#!/usr/bin/env zsh
# run_all.sh  放在项目根目录里直接 ./run_all.sh 即可

set -euo pipefail

DATA_ROOT="../data"
RULE_ROOT="../task_rules"

# 可选：自动创建 task_rules 目录
# mkdir -p "$RULE_ROOT"

# 遍历 data 下的一级目录（排除 .7z 文件）
for app_dir in "$DATA_ROOT"/*(/); do
    app_name="${app_dir:t}"           # bilibili、cloudmusic …
    mkdir -p "$RULE_ROOT/$app_name"   # 确保输出目录存在

    # 遍历该目录下的 type* 子目录
    for type_dir in "$app_dir"/type*(/); do
        type_name="${type_dir:t}"     # type1、type2 …
        out_file="$RULE_ROOT/$app_name/${app_name}-${type_name}.yaml"

        echo "→ Processing $type_dir  →  $out_file"
        python main.py "$type_dir" --output-file "$out_file"
    done
done

echo "All done!"