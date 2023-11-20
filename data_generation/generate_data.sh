START=$(date +%s.%N)
root pythia8.C
root exit
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
