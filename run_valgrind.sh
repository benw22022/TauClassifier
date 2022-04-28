valgrind --leak-check=full \
         --show-leak-kinds=all \
         --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         ../miniconda3/envs/tauid/bin/python3 tauclassifier.py train | tee valgrind_log.txt

