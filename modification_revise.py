import os
import re

def fix_seq_mods_in_mgf(root_dir, dry_run=False):
    """
    在 root_dir 下递归查找所有 .mgf 文件，只修改 SEQ= 行中的：
      1) 先去除所有括号: ()[]{}
      2) C 的修饰: +57.02*                -> +57.021
      3) N 的修饰: N+任意数字             -> N+0.984
      4) Q 的修饰: Q+任意数字             -> Q+0.984
      5) M 的修饰: M+15.99*               -> M+15.995
    """

    mgf_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".mgf"):
                mgf_files.append(os.path.join(dirpath, fname))

    print(f"Found {len(mgf_files)} .mgf files under {root_dir}")

    # 0) 去除括号: ()[]{}  （改为第一步执行）
    pattern_brackets = re.compile(r'[\(\)\[\]\{\}]')

    # 1) C 的修饰: +57.02... -> +57.021
    pattern_57 = re.compile(r"\+57\.0*2\d*")

    # 2) N 的修饰: N+任意数字 -> N+0.984
    pattern_N = re.compile(r"N\+\d*\.?\d+")

    # 3) Q 的修饰: Q+任意数字 -> Q+0.984
    pattern_Q = re.compile(r"Q\+\d*\.?\d+")

    # 4) M 的修饰: M+15.99* -> M+15.995
    pattern_M = re.compile(r"M\+15\.99\d*")

    for path in mgf_files:
        print(f"Processing: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        new_lines = []
        changed = False

        for line in lines:
            if line.startswith("SEQ="):
                old_line = line
                new_line = line

                # 第一步：先去除所有括号（这样后面的正则才能匹配到）
                new_line = pattern_brackets.sub('', new_line)

                # 第二步：替换修饰质量
                # C 的 57.02 系列
                new_line = pattern_57.sub("+57.021", new_line)

                # N 脱酰胺
                new_line = pattern_N.sub("N+0.984", new_line)

                # Q 脱酰胺
                new_line = pattern_Q.sub("Q+0.984", new_line)

                # M 氧化
                new_line = pattern_M.sub("M+15.995", new_line)

                if new_line != old_line:
                    changed = True
                    print("  Modified SEQ line:")
                    print("    OLD:", old_line.strip())
                    print("    NEW:", new_line.strip())

                new_lines.append(new_line)
            else:
                new_lines.append(line)

        if changed:
            if dry_run:
                print(f"  [dry_run] Changes detected but not written for {path}")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                print(f"  File updated: {path}")
        else:
            print("  No SEQ line with target patterns found (or nothing to change).")


if __name__ == "__main__":
    root_dir = "D:\data_spectrum\9species_Eng"

    # 第一步：dry-run，先看 OLD/NEW
    print("=" * 80)
    print("DRY RUN MODE - Preview changes only")
    print("=" * 80)
    fix_seq_mods_in_mgf(root_dir, dry_run=False)

    # 看完确认没问题后，把上一行注释掉，
    # 打开下面这几行真正写入：
    # print("\n" + "=" * 80)
    # print("WRITING CHANGES TO FILES")
    # print("=" * 80)
    # fix_seq_mods_in_mgf(root_dir, dry_run=False)