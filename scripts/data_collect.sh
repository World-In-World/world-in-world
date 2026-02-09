logfile="data_gen.log"
exec > >(stdbuf -o0 tee "$logfile") 2>&1
xvfb-run -a -s "-screen 0 1024x768x24" python wiw_manip/envs/tools/dataset_generator_NLP.py
