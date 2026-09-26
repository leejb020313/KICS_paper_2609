f=$1; o=${f%.txt}.out
[ -s "$o" ] && exit 0
claude -p "$(cat $f)" --model sonnet --output-format text > $o 2>&1
