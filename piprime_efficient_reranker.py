#!/usr/bin/env python
"""
PiPrime专用的高效重排序器
不依赖Casanovo，只使用外部提供的spectrum embedding
"""

import pickle
import numpy as np
import pandas as pd
import torch
import requests
import re
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from pyteomics import mgf, mass

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PiPrimeEfficientReranker:
    """PiPrime专用的高效重排序器（使用PiPrime encoder）"""

    def __init__(self, piprime_model, piprime_config, koina_url="https://koina.wilhelmlab.org"):
        """
        初始化重排序器
        
        Parameters:
        -----------
        piprime_model : Spec2Pep
            PiPrime模型（用于编码spectra）
        piprime_config : dict
            PiPrime配置
        koina_url : str
            Koina服务器URL
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        
        self.piprime_model = piprime_model
        self.piprime_config = piprime_config
        self.koina_url = koina_url
        self.encoding_cache = {}  # 缓存
        self.precomputed_index = None  # 预计算索引

    def convert_piprime_to_unimod(self, peptide):
        """
        将PiPrime格式的修饰转换为UNIMOD格式
        
        PiPrime格式: C+57.021, M+15.995, N+0.984
        UNIMOD格式: C[UNIMOD:4], M[UNIMOD:35], N[UNIMOD:7]
        """
        # 质量到UNIMOD ID的映射
        mass_to_unimod = [
            (56.87, 57.17, 'UNIMOD:4'),   # Carbamidomethyl (C)
            (15.85, 16.15, 'UNIMOD:35'),  # Oxidation (M)
            (0.83, 1.13, 'UNIMOD:7'),     # Deamidation (N/Q)
            (41.86, 42.16, 'UNIMOD:1'),   # Acetyl (N-term)
            (42.86, 43.16, 'UNIMOD:5'),   # Carbamyl (N-term)
        ]
        
        # 匹配PiPrime格式的修饰: C+57.021
        pattern = r'([A-Z])([\+\-][\d\.]+)'
        
        def replace_mod(match):
            aa = match.group(1)
            mod_str = match.group(2)
            try:
                mod_mass = float(mod_str)
                # 查找匹配的UNIMOD ID
                for min_mass, max_mass, unimod_id in mass_to_unimod:
                    if min_mass <= mod_mass <= max_mass:
                        return f'{aa}[{unimod_id}]'
                # 未知修饰，保留质量
                return f'{aa}[{mod_str}]'
            except ValueError:
                return match.group(0)
        
        converted = re.sub(pattern, replace_mod, peptide)
        # L替换为I
        converted = converted.replace('L', 'I')
        return converted
    
    def normalize_peptide(self, peptide):
        """
        标准化肽段 - 去除修饰并将L替换为I
        
        支持的修饰格式:
        - PiPrime格式: C+57.021, M+15.995
        - 括号格式: C[+57.021], M[+15.995]
        - UNIMOD格式: C[UNIMOD:4]
        """
        # 去除括号修饰 [+57.021] 或 [UNIMOD:4]
        clean_seq = re.sub(r'\[.*?\]', '', peptide)
        # 去除PiPrime格式修饰 +57.021 或 -17.027
        clean_seq = re.sub(r'[\+\-][\d\.]+', '', clean_seq)
        # 去除N端修饰前缀（如果有）
        clean_seq = re.sub(r'^[\[\]A-Za-z0-9\-\+\.]+\-', '', clean_seq)
        # L替换为I
        clean_seq = clean_seq.replace('L', 'I')
        return clean_seq
    
    def calculate_peptide_mass_with_mods(self, peptide, charge=1):
        """
        计算带修饰的肽段质量
        
        支持格式:
        - UNIMOD格式: M[UNIMOD:35]PEPTIDE
        - 质量格式: M[+15.994915]PEPTIDE
        - 括号格式: M(+15.99)PEPTIDE
        
        Returns:
        --------
        precursor_mz : float
            计算的m/z值
        """
        # Unimod ID到质量的映射
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
        
        # 分离氨基酸序列和修饰
        total_mod_mass = 0.0
        clean_seq = ""
        
        i = 0
        while i < len(peptide):
            if peptide[i] in '[(':
                # 找到修饰的结束位置
                end_char = ']' if peptide[i] == '[' else ')'
                end = peptide.find(end_char, i)
                if end == -1:
                    break
                
                mod_str = peptide[i+1:end]
                
                # 解析修饰
                if mod_str.startswith('UNIMOD:'):
                    # UNIMOD格式
                    if mod_str in unimod_masses:
                        total_mod_mass += unimod_masses[mod_str]
                    else:
                        logger.warning(f"Unknown UNIMOD: {mod_str}")
                else:
                    # 数值格式: +15.994915 或 15.994915 或 +.98
                    try:
                        mod_mass = float(mod_str.replace('+', ''))
                        total_mod_mass += mod_mass
                    except ValueError:
                        logger.warning(f"Cannot parse modification: {mod_str}")
                
                i = end + 1
            else:
                # 普通氨基酸
                if peptide[i].isalpha():
                    clean_seq += peptide[i]
                i += 1
        
        # 计算基础肽段质量（不含修饰）
        try:
            base_mass = mass.calculate_mass(sequence=clean_seq, charge=0)
            # 加上修饰质量
            total_mass = base_mass + total_mod_mass
            # 计算m/z
            precursor_mz = (total_mass + charge * 1.007276) / charge
            return precursor_mz
        except Exception as e:
            logger.warning(f"Failed to calculate mass for {peptide}: {e}")
            # 回退到不含修饰的计算
            return mass.calculate_mass(sequence=clean_seq, charge=charge)

    def load_precomputed_index(self, index_file):
        """加载预计算索引"""
        logger.info(f"Loading precomputed index: {index_file}")
        with open(index_file, 'rb') as f:
            self.precomputed_index = pickle.load(f)
        logger.info(f"Loaded {len(self.precomputed_index)} unique sequences")
        return self.precomputed_index
    
    def encode_spectrum_from_arrays(self, mz_array, intensity_array, precursor_mz, precursor_charge):
        """从m/z和intensity数组编码spectrum（使用PiPrime encoder）"""
        # 使用PiPrime的预处理方法
        from piprime_reranker import process_peaks
        
        peaks = process_peaks(
            mz_array, intensity_array,
            precursor_mz, precursor_charge,
            self.piprime_config
        )
        peaks = peaks.to(self.device)
        
        # 使用PiPrime encoder
        with torch.no_grad():
            enc_out, enc_mask = self.piprime_model.encoder(peaks.unsqueeze(0))
            embedding = enc_out.mean(dim=1).cpu().numpy()[0]
        
        return embedding

    def generate_prosit_spectrum(self, peptide, charge, fragmentation_type="HCD"):
        """使用 Prosit_2025_intensity_MultiFrag 预测谱图"""

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
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=30)
            if response.status_code != 200:
                return None

            result = response.json()

            # MultiFrag返回3个输出
            intensities_data = None
            mz_data = None

            for output in result.get('outputs', []):
                output_name = output.get('name')
                if output_name == 'intensities':
                    intensities_data = output.get('data', [])
                elif output_name == 'mz':
                    mz_data = output.get('data', [])

            if not intensities_data or not mz_data:
                return None

            # 解析为峰列表
            mz_list, intensity_list = self._prosit_to_peaks(peptide, charge, intensities_data)

            if len(mz_list) == 0:
                return None

            # 使用带修饰的质量计算
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
        """将Prosit输出转为峰列表（简化版，不处理修饰）"""
        mz_list, intensity_list = [], []
        
        # 提取不含修饰的序列长度
        clean_seq = self.normalize_peptide(sequence)
        peptide_len = len(clean_seq)
        idx = 0

        # b ions
        for ion_charge in [1, 2, 3]:
            for pos in range(1, peptide_len):
                fragment = clean_seq[:pos]
                
                for mod in ['', '-H2O', '-NH3']:
                    if idx < len(intensities):
                        intensity = intensities[idx]
                        if intensity > 0:
                            b_mz = mass.fast_mass(fragment, ion_type='b', charge=ion_charge)
                            if mod == '-H2O':
                                b_mz -= 18.01056 / ion_charge
                            elif mod == '-NH3':
                                b_mz -= 17.02655 / ion_charge
                            mz_list.append(b_mz)
                            intensity_list.append(max(float(intensity), 0.00001))
                    idx += 1

        # y ions
        for ion_charge in [1, 2, 3]:
            for pos in range(1, peptide_len):
                fragment = clean_seq[-pos:]
                
                for mod in ['', '-H2O', '-NH3']:
                    if idx < len(intensities):
                        intensity = intensities[idx]
                        if intensity > 0:
                            y_mz = mass.fast_mass(fragment, ion_type='y', charge=ion_charge)
                            if mod == '-H2O':
                                y_mz -= 18.01056 / ion_charge
                            elif mod == '-NH3':
                                y_mz -= 17.02655 / ion_charge
                            mz_list.append(y_mz)
                            intensity_list.append(max(float(intensity), 0.00001))
                    idx += 1

        return mz_list, intensity_list
    
    def rerank_with_external_embedding(
        self,
        query_embedding,
        candidates,
        precursor_mz,
        precursor_charge,
        use_prosit=True,
        top_k=3
    ):
        """
        使用外部提供的spectrum embedding进行重排序
        
        Parameters:
        -----------
        query_embedding : np.ndarray
            外部提供的query spectrum embedding（来自PiPrime encoder）
        candidates : list of dict
            候选peptide列表，每个dict包含 'peptide' 和 'score'
        precursor_mz : float
            Precursor m/z
        precursor_charge : int
            Precursor电荷
        use_prosit : bool
            是否使用Prosit预测（默认True）
        top_k : int
            取Top-K个相似度的平均值（默认3）
            
        Returns:
        --------
        dict : 重排序后的Top-1结果
        """
        if self.precomputed_index is None:
            raise ValueError("Precomputed index not loaded. Call load_precomputed_index() first.")
        
        results = []
        
        # 用于缓存已计算的peptide的similarity
        similarity_cache = {}
        
        for candidate in candidates:
            peptide = candidate['peptide']
            denovo_score = candidate['score']
            
            # 检查缓存：如果这个peptide已经计算过，直接使用缓存的结果
            if peptide in similarity_cache:
                cached_result = similarity_cache[peptide]
                results.append({
                    'peptide': peptide,
                    'denovo_score': denovo_score,
                    'similarity': cached_result['similarity'],
                    'matched_count': cached_result.get('matched_count', 0),
                    'source': cached_result['source']
                })
                continue
            
            # 转换PiPrime格式到UNIMOD格式
            unimod_peptide = self.convert_piprime_to_unimod(peptide)
            clean_peptide = self.normalize_peptide(peptide)
            
            # O(1) 查找数据库
            # 尝试顺序: 1) UNIMOD格式（带修饰） 2) 原始格式 3) 标准化格式（去除修饰）
            ref_spectra = None
            if unimod_peptide in self.precomputed_index:
                ref_spectra = self.precomputed_index[unimod_peptide]["spectra"]
            elif peptide in self.precomputed_index:
                ref_spectra = self.precomputed_index[peptide]["spectra"]
            elif clean_peptide in self.precomputed_index:
                ref_spectra = self.precomputed_index[clean_peptide]["spectra"]
            
            if ref_spectra is not None:
                
                # 限制到前10个
                ref_spectra = ref_spectra[:10]
                
                # 计算与每个参考谱图的相似度
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
                
                # 取Top-K相似度的平均值
                top_k_similarities = sorted(similarities, reverse=True)[:top_k]
                final_similarity = np.mean(top_k_similarities)
                
                result_dict = {
                    'peptide': peptide,
                    'denovo_score': denovo_score,
                    'similarity': final_similarity,
                    'matched_count': len(ref_spectra),
                    'all_similarities': similarities,
                    'top_k_similarities': top_k_similarities,
                    'source': 'Database'
                }
                results.append(result_dict)
                
                # 缓存结果（不包括all_similarities和top_k_similarities以节省内存）
                similarity_cache[peptide] = {
                    'similarity': final_similarity,
                    'matched_count': len(ref_spectra),
                    'source': 'Database'
                }
            
            elif use_prosit:
                # 使用Prosit_2025_MultiFrag预测
                # 注意：Prosit不支持修饰，需要使用去除修饰后的peptide
                prosit_peptide = clean_peptide  # 使用去除修饰后的序列
                prosit_spec = self.generate_prosit_spectrum(
                    prosit_peptide,
                    precursor_charge,
                    fragmentation_type='HCD'
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
                    
                    result_dict = {
                        'peptide': peptide,
                        'denovo_score': denovo_score,
                        'similarity': similarity,
                        'matched_count': 0,
                        'source': 'Prosit'
                    }
                    results.append(result_dict)
                    
                    # 缓存Prosit结果
                    similarity_cache[peptide] = {
                        'similarity': similarity,
                        'matched_count': 0,
                        'source': 'Prosit'
                    }
                else:
                    result_dict = {
                        'peptide': peptide,
                        'denovo_score': denovo_score,
                        'similarity': -1.0,
                        'matched_count': 0,
                        'source': 'Failed'
                    }
                    results.append(result_dict)
                    
                    # 缓存Failed结果
                    similarity_cache[peptide] = {
                        'similarity': -1.0,
                        'matched_count': 0,
                        'source': 'Failed'
                    }
            else:
                # 不使用Prosit
                result_dict = {
                    'peptide': peptide,
                    'denovo_score': denovo_score,
                    'similarity': -1.0,
                    'matched_count': 0,
                    'source': 'NotFound'
                }
                results.append(result_dict)
                
                # 缓存NotFound结果
                similarity_cache[peptide] = {
                    'similarity': -1.0,
                    'matched_count': 0,
                    'source': 'NotFound'
                }
        
        # 按相似度排序
        try:
            results_df = pd.DataFrame(results)
            if len(results_df) == 0:
                return {
                    'peptide': '',
                    'similarity': -1.0,
                    'denovo_score': 0.0,
                    'rerank': -1,
                    'source': 'NoResults'
                }, {}, {}
            
            # 转换similarity为数值类型
            results_df['similarity'] = pd.to_numeric(results_df['similarity'], errors='coerce')
            results_df = results_df.sort_values('similarity', ascending=False).reset_index(drop=True)
            results_df['rerank'] = range(1, len(results_df) + 1)
            
            # 创建peptide到similarity和source的映射
            similarity_dict = dict(zip(results_df['peptide'], results_df['similarity']))
            source_dict = dict(zip(results_df['peptide'], results_df['source']))
            
            # 返回Top-1结果、所有similarity和source
            if len(results_df) > 0:
                top_result = results_df.iloc[0].to_dict()
                return top_result, similarity_dict, source_dict
            else:
                return {
                    'peptide': '',
                    'similarity': -1.0,
                    'denovo_score': 0.0,
                    'rerank': -1,
                    'source': 'EmptyResults'
                }, {}, {}
                
        except Exception as e:
            print(f"Error in reranking results: {e}")
            # 返回De Novo最高分的候选
            if candidates:
                top_candidate = max(candidates, key=lambda x: x['score'])
                return {
                    'peptide': top_candidate['peptide'],
                    'similarity': -1.0,
                    'denovo_score': top_candidate['score'],
                    'rerank': 1,
                    'source': 'DeNovoFallback'
                }, {}, {}
            else:
                return {
                    'peptide': '',
                    'similarity': -1.0,
                    'denovo_score': 0.0,
                    'rerank': -1,
                    'source': 'Error'
                }, {}, {}