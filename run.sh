for p in 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1; do
python main_graph_classification_mp_timing.py --model amp --dataset PTC --processes=$p
#sbatch --cpus-per-task=$(($p+1)) runany.sh main_graph_classification_mp_timing.py --model amp --dataset PTC --processes=$p
#sbatch --cpus-per-task=$(($p+1)) runany.sh main_graph_classification_mp_timing.py --model amp --dataset GINPTC --processes=$p
#sbatch --cpus-per-task=$(($p+1)) runany.sh main_graph_classification_mp_timing.py --model amp --dataset PROTEINS --processes=$p
#sbatch --cpus-per-task=$(($p+1)) runany.sh main_graph_classification_mp_timing.py --model amp --dataset IMDB-BINARY --processes=$p
#sbatch --cpus-per-task=$(($p+1)) runany.sh main_graph_classification_mp_timing.py --model amp --dataset IMDB-MULTI --processes=$p
done
