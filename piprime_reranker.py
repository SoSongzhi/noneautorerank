#!/usr/bin/env python
"""
使用PiPrime的高效重排序器
基于efficient_reranker.py的逻辑，但使用PiPrime非自回归模型

特点:
1. 使用PiPrime CTC模型生成候选peptide
2. 支持大beam size (100条候选)
3. 预建索引，快速检索
4. 计算Top-3最高相似度的平均值
5. 混合策略：Database优先，Prosit fallback
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import requests
import urllib3
import re
import logging

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import yaml
import os
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from pyteomics import mgf, mass

# 添加PiPrime路径 - 使用当前目录
sys.path.insert(0, os.path.dirname(__file__))

from PrimeNovo.denovo.model import Spec2Pep

try:
    import spectrum_utils.spectrum as sus
except ImportError:
    print("警告: spectrum_utils未安装，将使用简化的谱图预处理")
    sus = None

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_piprime_config():
    """加载PiPrime配置"""
    config_path = os.path.join(os.path.dirname(__file__), "PrimeNovo", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        # 默认配置
        return {
            "min_mz": 50.0,
            "max_mz": 2500.0,
            "remove_precursor_tol": 2.0,
            "min_intensity": 0.01,
            "n_peaks": 200,
            "max_length": 30
        }


def process_peaks(mz_array, int_array, precursor_mz, precursor_charge, config):
    """预处理质谱峰数据（使用spectrum_utils）"""
    if sus is None:
        # Fallback: 简单处理
        valid_idx = int_array > 0
        mz_array = mz_array[valid_idx]
        int_array = int_array[valid_idx]
        
        sorted_idx = np.argsort(mz_array)
        mz_array = mz_array[sorted_idx]
        int_array = int_array[sorted_idx]
        
        if len(int_array) > 0:
            int_array = int_array / np.max(int_array)
        
        if len(mz_array) > config["n_peaks"]:
            top_idx = np.argsort(int_array)[-config["n_peaks"]:]
            top_idx = np.sort(top_idx)
            mz_array = mz_array[top_idx]
            int_array = int_array[top_idx]
        
        return torch.tensor(np.array([mz_array, int_array])).T.float()
    
    # 使用spectrum_utils
    spectrum = sus.MsmsSpectrum(
        "", precursor_mz, precursor_charge, 
        mz_array.astype(np.float64), 
        int_array.astype(np.float32)
    )
    try:
        spectrum.set_mz_range(config["min_mz"], config["max_mz"])
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.remove_precursor_peak(config["remove_precursor_tol"], "Da")
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.filter_intensity(config["min_intensity"], config["n_peaks"])
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.scale_intensity("root", 1)
        intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)
        return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
    except ValueError:
        return torch.tensor([[0, 1]]).float()


class PiPrimeReranker:
    """使用PiPrime的高效重排序器"""

    def __init__(self, model_path=None, config_path=None, koina_url="https://koina.wilhelmlab.org", beam_size=100):
        """
        初始化重排序器

        Parameters:
        -----------
        model_path : str, optional
            PiPrime模型checkpoint路径
        config_path : str, optional
            配置文件路径
        koina_url : str
            Koina服务器URL
        beam_size : int
            Beam search宽度（默认100）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = load_piprime_config()
        
        logger.info(f"Config loaded: n_peaks={self.config['n_peaks']}")

        # 设置默认模型路径
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "model_massive.ckpt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PiPrime model not found: {model_path}")

        logger.info(f"Loading PiPrime model: {model_path}")
        
        # 加载PiPrime模型
        self.model = Spec2Pep.load_from_checkpoint(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        logger.info("✅ PiPrime model loaded!")

        # 使用模型内置的CTC Decoder
        self.beam_size = beam_size
        self.decoder = self.model.ctc_decoder
        logger.info(f"✅ CTC Decoder ready (beam_size: {beam_size})")

        self.koina_url = koina_url
        self.encoding_cache = {}  # 编码缓存
        self.precomputed_index = None  # 预计算索引

    def normalize_peptide(self, peptide):
        """标准化肽段 - 去除修饰并将L替换为I"""
        clean_seq = re.sub(r'\[.*?\]', '', peptide)
        clean_seq = re.sub(r'^[\[\]A-Za-z0-9\-\+\.]+\-', '', clean_seq)
        clean_seq = clean_seq.replace('L', 'I')
        return clean_seq
    
    def calculate_peptide_mass_with_mods(self, peptide, charge=1):
        """
        计算带修饰的肽段质量
        
        支持格式:
        - UNIMOD格式: M[UNIMOD:35]PEPTIDE
        - 质量格式: M[+15.994915]PEPTIDE
        - 括号格式: M(+15.99)PEPTIDE
        """
        unimod_masses = {
            'UNIMOD:35': 15.994915,   # Oxidation (M)
            'UNIMOD:4': 57.021464,    # Carbamidomethyl (C)
            'UNIMOD:7': 0.984016,     # Deamidation (N/Q)
            'UNIMOD:1': 42.010565,    # Acetyl (N-term)
            'UNIMOD:5': 43.005814,    # Carbamyl (N-term)
            'UNIMOD:28': -17.026549,  # Gln->pyro-Glu (Q)
            'UNIMOD:27': -18.010565,  # Glu->pyro-Glu (E)
            'UNIMOD:385': -17.026549, # Ammonia-loss (N-term)
            'UNIMOD:21': 79.966331,   # Phospho (S/T/Y)
            'UNIMOD:34': 14.015650,   # Methyl
        }
        
        total_mod_mass = 0.0
        clean_seq = ""
        
        i = 0
        while i < len(peptide):
            if peptide[i] in '[(':
                end_char = ']' if peptide[i] == '[' else ')'
                end = peptide.find(end_char, i)
                if end == -1:
                    break
                
                mod_str = peptide[i+1:end]
                
                if mod_str.startswith('UNIMOD:'):
                    if mod_str in unimod_masses:
                        total_mod_mass += unimod_masses[mod_str]
                    else:
                        logger.warning(f"Unknown UNIMOD: {mod_str}")
                else:
                    try:
                        mod_mass = float(mod_str.replace('+', ''))
                        total_mod_mass += mod_mass
                    except ValueError:
                        logger.warning(f"Cannot parse modification: {mod_str}")
                
                i = end + 1
            else:
                if peptide[i].isalpha():
                    clean_seq += peptide[i]
                i += 1
        
        try:
            base_mass = mass.calculate_mass(sequence=clean_seq, charge=0)
            total_mass = base_mass + total_mod_mass
            precursor_mz = (total_mass + charge * 1.007276) / charge
            return precursor_mz
        except Exception as e:
            logger.warning(f"Failed to calculate mass for {peptide}: {e}")
            return mass.calculate_mass(sequence=clean_seq, charge=charge)

    def load_precomputed_index(self, index_file):
        """加载预计算索引"""
        logger.info(f"Loading precomputed index: {index_file}")
        with open(index_file, 'rb') as f:
            self.precomputed_index = pickle.load(f)
        logger.info(f"Loaded {len(self.precomputed_index)} unique sequences")
        return self.precomputed_index

    def generate_candidates_piprime(self, mz_array, intensity_array, precursor_mz, charge):
        """
        使用PiPrime生成候选peptide
        
        Returns: list of {'peptide': str, 'score': float}
        """
        # 预处理谱图
        peaks = process_peaks(mz_array, intensity_array, precursor_mz, charge, self.config)
        peaks = peaks.to(self.device)
        
        # 准备precursors
        precursor_mass = (precursor_mz - 1.007276) * charge
        precursors = torch.tensor([[precursor_mass, charge, precursor_mz]], 
                                  dtype=torch.float32, device=self.device)
        
        # PiPrime前向传播
        with torch.no_grad():
            # Encoder
            enc_out, enc_mask = self.model.encoder(peaks.unsqueeze(0))
            
            # Decoder - 获取概率矩阵
            output_logits, _, _ = self.model.decoder(None, precursors, enc_out, enc_mask)
            
            # Log Softmax
            log_probs = F.log_softmax(output_logits, dim=-1)
        
        # CTC Beam Search - 在概率矩阵上操作
        tokens_k, scores_k = self.decoder.decode_topk(log_probs, topk=self.beam_size)
        
        # 转换为peptide序列
        results = []
        tokens_k = tokens_k[0]  # 取第一个batch (B=1)
        scores_k = scores_k[0]  # 取第一个batch
        
        for idx in range(len(scores_k)):
            token_seq = tokens_k[idx].cpu().numpy()
            log_prob = scores_k[idx].item()
            
            # 过滤padding（0 tokens）
            token_seq = token_seq[token_seq > 0]
            
            if len(token_seq) > 0:
                # 转换为氨基酸序列
                aa_seq = [self.model.decoder._idx2aa.get(int(t), '') for t in token_seq]
                peptide = "".join(aa_seq)
                
                if peptide:  # 过滤空序列
                    results.append({
                        'peptide': peptide,
                        'score': np.exp(log_prob)  # 转为概率
                    })
        
        return results

    def encode_spectrum_from_arrays(self, mz_array, intensity_array, precursor_mz, precursor_charge):
        """使用PiPrime Encoder编码谱图为向量"""
        # 预处理
        peaks = process_peaks(mz_array, intensity_array, precursor_mz, precursor_charge, self.config)
        peaks = peaks.to(self.device)
        
        with torch.no_grad():
            enc_out, _ = self.model.encoder(peaks.unsqueeze(0))
            # 平均池化
            embedding = enc_out.mean(dim=1).cpu().numpy()[0]
        
        return embedding

    def get_spectrum_from_mgf(self, mgf_file, spec_index):
        """从MGF文件获取指定索引的谱图"""
        with mgf.MGF(mgf_file) as reader:
            for idx, spec in enumerate(reader):
                if idx == spec_index:
                    pepmass = spec['params'].get('pepmass', [0])
                    if isinstance(pepmass, (list, tuple)):
                        precursor_mz = pepmass[0] if len(pepmass) > 0 else 0
                    else:
                        precursor_mz = pepmass if pepmass else 0

                    charge = spec['params'].get('charge', [2])
                    if isinstance(charge, (list, tuple)):
                        charge = charge[0] if len(charge) > 0 else 2
                    else:
                        charge = charge if charge else 2

                    if isinstance(charge, str):
                        charge = int(charge.replace('+', '').replace('-', ''))
                    else:
                        charge = int(charge)

                    fragmentation_type = "HCD"

                    if 'fragmentation' in spec['params']:
                        frag = spec['params']['fragmentation'].upper()
                        if 'CID' in frag:
                            fragmentation_type = "CID"
                        elif 'HCD' in frag:
                            fragmentation_type = "HCD"
                        elif 'ECD' in frag:
                            fragmentation_type = "ECD"
                        elif 'EID' in frag:
                            fragmentation_type = "EID"
                        elif 'UVPD' in frag:
                            fragmentation_type = "UVPD"
                        elif 'ETCID' in frag or 'ETciD' in frag:
                            fragmentation_type = "ETciD"

                    return {
                        'mz': spec['m/z array'],
                        'intensity': spec['intensity array'],
                        'precursor_mz': float(precursor_mz),
                        'charge': charge,
                        'fragmentation_type': fragmentation_type
                    }
        return None

    def generate_prosit_spectrum(self, peptide, charge, fragmentation_type="HCD", query_index=None):
        """使用 Prosit_2025_intensity_MultiFrag 生成理论谱图"""

        url = f"{self.koina_url}/v2/models/Prosit_2025_intensity_MultiFrag/infer"
        payload = {
            "id": "prosit_prediction",
            "inputs": [
                {
                    "name": "peptide_sequences",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [peptide]
                },
                {
                    "name": "precursor_charges",
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [int(charge)]
                },
                {
                    "name": "fragmentation_types",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [fragmentation_type]
                }
            ]
        }

        try:
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=30, verify=False)
            if response.status_code != 200:
                return None

            result = response.json()

            intensities_data = None
            for output in result.get('outputs', []):
                output_name = output.get('name')
                if output_name == 'intensities':
                    intensities_data = output.get('data', [])
                    break

            if not intensities_data:
                return None

            mz_list, intensity_list = self._prosit_to_peaks(peptide, charge, intensities_data)

            if len(mz_list) == 0:
                return None

            precursor_mz = self.calculate_peptide_mass_with_mods(peptide, charge)

            return {
                'mz': np.array(mz_list),
                'intensity': np.array(intensity_list),
                'precursor_mz': precursor_mz,
                'charge': charge
            }
        except Exception as e:
            logger.warning(f"Prosit failed for {peptide}: {e}")
            return None

    def _prosit_to_peaks(self, sequence, charge, intensities):
        """将Prosit输出转为峰列表"""
        mz_list, intensity_list = [], []
        
        clean_seq = self.normalize_peptide(sequence)
        peptide_len = len(clean_seq)
        idx = 0

        # b ions
        for ion_charge in [1, 2, 3]:
            for pos in range(1, peptide_len):
                fragment = sequence[:pos]
                
                for mod in ['', '-H2O', '-NH3']:
                    if idx < len(intensities):
                        intensity = intensities[idx]
                        if intensity > 0:
                            try:
                                b_mz = mass.fast_mass(self.normalize_peptide(fragment), ion_type='b', charge=ion_charge)
                                if mod == '-H2O':
                                    b_mz -= 18.01056 / ion_charge
                                elif mod == '-NH3':
                                    b_mz -= 17.02655 / ion_charge
                                mz_list.append(b_mz)
                                intensity_list.append(max(float(intensity), 0.00001))
                            except:
                                pass
                    idx += 1

        # y ions
        for ion_charge in [1, 2, 3]:
            for pos in range(1, peptide_len):
                fragment = sequence[-pos:]
                
                for mod in ['', '-H2O', '-NH3']:
                    if idx < len(intensities):
                        intensity = intensities[idx]
                        if intensity > 0:
                            try:
                                y_mz = mass.fast_mass(self.normalize_peptide(fragment), ion_type='y', charge=ion_charge)
                                if mod == '-H2O':
                                    y_mz -= 18.01056 / ion_charge
                                elif mod == '-NH3':
                                    y_mz -= 17.02655 / ion_charge
                                mz_list.append(y_mz)
                                intensity_list.append(max(float(intensity), 0.00001))
                            except:
                                pass
                    idx += 1

        return mz_list, intensity_list

    def rerank_with_efficient_index(
        self,
        query_mgf,
        query_index,
        candidates,
        use_prosit=True,
        top_k=3
    ):
        """
        使用高效索引重排序
        
        与efficient_reranker.py逻辑完全相同，但使用PiPrime生成候选

        Parameters:
        - query_mgf: 查询谱图MGF
        - query_index: 谱图索引
        - candidates: list of dict with 'peptide' and 'score'
        - use_prosit: 是否使用Prosit fallback
        - top_k: 取前k个最高相似度的平均值（默认3）
        """
        if self.precomputed_index is None:
            raise ValueError("Precomputed index not loaded. Call load_precomputed_index() first.")

        # 1. 编码查询谱图
        query_spec = self.get_spectrum_from_mgf(query_mgf, query_index)
        if query_spec is None:
            raise ValueError(f"Failed to load spectrum at index {query_index}")

        query_embedding = self.encode_spectrum_from_arrays(
            query_spec['mz'], query_spec['intensity'],
            query_spec['precursor_mz'], query_spec['charge']
        )

        # 2. 对每个候选计算相似度
        results = []

        for candidate in candidates:
            peptide = candidate['peptide']
            denovo_score = candidate['score']
            clean_peptide = self.normalize_peptide(peptide)

            # O(1) 查找
            if clean_peptide in self.precomputed_index:
                ref_spectra = self.precomputed_index[clean_peptide]["spectra"]

                # 限制到前10个
                ref_spectra = ref_spectra[:10]

                # 计算每个参考谱图的相似度
                similarities = []
                for ref_spec in ref_spectra:
                    # 检查缓存
                    cache_key = f"ref_{ref_spec['index']}"
                    if cache_key in self.encoding_cache:
                        ref_embedding = self.encoding_cache[cache_key]
                    else:
                        ref_embedding = self.encode_spectrum_from_arrays(
                            ref_spec['mz'], ref_spec['intensity'],
                            ref_spec['precursor_mz'], ref_spec['charge']
                        )
                        self.encoding_cache[cache_key] = ref_embedding

                    sim = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        ref_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)

                # 取Top-K最高相似度的平均值
                top_k_similarities = sorted(similarities, reverse=True)[:top_k]
                final_similarity = np.mean(top_k_similarities)

                results.append({
                    'peptide': peptide,
                    'denovo_score': denovo_score,
                    'similarity': final_similarity,
                    'matched_count': len(ref_spectra),
                    'all_similarities': similarities,
                    'top_k_similarities': top_k_similarities,
                    'source': 'Database'
                })

            elif use_prosit:
                # 使用Prosit_2025_MultiFrag
                prosit_spec = self.generate_prosit_spectrum(
                    peptide,
                    query_spec['charge'],
                    fragmentation_type=query_spec.get('fragmentation_type', 'HCD'),
                    query_index=query_index
                )

                if prosit_spec:
                    prosit_embedding = self.encode_spectrum_from_arrays(
                        prosit_spec['mz'], prosit_spec['intensity'],
                        prosit_spec['precursor_mz'], prosit_spec['charge']
                    )

                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        prosit_embedding.reshape(1, -1)
                    )[0][0]

                    results.append({
                        'peptide': peptide,
                        'denovo_score': denovo_score,
                        'similarity': similarity,
                        'matched_count': 0,
                        'source': 'Prosit'
                    })
                else:
                    results.append({
                        'peptide': peptide,
                        'denovo_score': denovo_score,
                        'similarity': -1.0,
                        'matched_count': 0,
                        'source': 'Failed'
                    })
            else:
                results.append({
                    'peptide': peptide,
                    'denovo_score': denovo_score,
                    'similarity': -1.0,
                    'matched_count': 0,
                    'source': 'NotFound'
                })

        # 3. 按相似度排序
        try:
            results_df = pd.DataFrame(results)
            if len(results_df) == 0:
                return {
                    'peptide': '',
                    'similarity': -1.0,
                    'denovo_score': 0.0,
                    'rerank': -1,
                    'source': 'NoResults'
                }
            
            results_df['similarity'] = pd.to_numeric(results_df['similarity'], errors='coerce')
            results_df = results_df.sort_values('similarity', ascending=False).reset_index(drop=True)
            results_df['rerank'] = range(1, len(results_df) + 1)

            if len(results_df) > 0:
                top_result = results_df.iloc[0].to_dict()
                return top_result
            else:
                return {
                    'peptide': '',
                    'similarity': -1.0,
                    'denovo_score': 0.0,
                    'rerank': -1,
                    'source': 'EmptyResults'
                }
                
        except Exception as e:
            print(f"Error in reranking results: {e}")
            if candidates:
                top_candidate = max(candidates, key=lambda x: x['score'])
                return {
                    'peptide': top_candidate['peptide'],
                    'similarity': -1.0,
                    'denovo_score': top_candidate['score'],
                    'rerank': 1,
                    'source': 'DeNovoFallback'
                }
            else:
                return {
                    'peptide': '',
                    'similarity': -1.0,
                    'denovo_score': 0.0,
                    'rerank': -1,
                    'source': 'Error'
                }