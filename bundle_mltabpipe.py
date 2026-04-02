import os
import re

def bundle_pipeline(package_name='mltabpipe', output_file='mltab_bundle.py'):
    """Bundles all components of the mltabpipe library into one file safely."""
    
    # Order of bundling to ensure dependencies are met
    bundle_order = [
        'core/common.py',
        'preprocessing/features.py',
        'ensemble/gbdt.py',
        'ensemble/rf.py',
        'ensemble/ridge.py',
        'ensemble/ydf.py',
        'ensemble/te_logit.py',
        'ensemble/stacker.py',
    ]
    
    bundled_code = []
    all_imports = set()
    
    print(f"--- Bundling {package_name} into {output_file} ---")
    
    for rel_path in bundle_order:
        abs_path = os.path.join(package_name, rel_path)
        if not os.path.exists(abs_path):
            continue
            
        print(f"  Adding {rel_path}...")
        with open(abs_path, 'r') as f:
            content = f.read()
            
        lines = content.splitlines()
        module_lines = []
        skip_lines = 0
        
        for i, line in enumerate(lines):
            if skip_lines > 0:
                skip_lines -= 1
                continue
            
            # 1. Skip internal/relative imports completely
            internal_import_regex = r'^\s*(from\s+\.|import\s+\.|from\s+' + package_name + r'|import\s+' + package_name + r')'
            if re.match(internal_import_regex, line):
                if '(' in line and ')' not in line:
                    for j in range(i + 1, len(lines)):
                        if ')' in lines[j]:
                            skip_lines = j - i
                            break
                continue

            # 2. Collect only TOP-LEVEL external imports to move to top
            # Indented imports (inside try/if/def) must stay in place!
            if re.match(r'^(import\s+|from\s+\w+\s+import)', line):
                full_import = line
                if '(' in line and ')' not in line:
                    for j in range(i+1, len(lines)):
                        full_import += "\n" + lines[j]
                        skip_lines = j - i
                        if ')' in lines[j]: break
                all_imports.add(full_import.strip())
                continue
                
            module_lines.append(line)
            
        bundled_code.append(f"\n# {'='*20} FROM {rel_path} {'='*20}\n")
        bundled_code.append("\n".join(module_lines))

    # Final Construction
    final_output = [
        "# " + "="*60,
        "# MLTabPipe Standalone Bundle for Kaggle",
        "# Generated at: " + os.popen('date').read().strip(),
        "# " + "="*60 + "\n",
        "\n".join(sorted(list(all_imports))),
        "\n"
    ]
    final_output.extend(bundled_code)
    
    with open(output_file, 'w') as f:
        f.write("\n".join(final_output))
    
    print(f"\nSUCCESS: Bundle created at {output_file}")

if __name__ == "__main__":
    bundle_pipeline()
