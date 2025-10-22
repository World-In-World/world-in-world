# this script is used to set the environment variables for the inference scripts
# feel free to modify this script to set the environment variables for your own inference scripts

export HF_TOKEN="XXX"


# if HF_TOKEN is "XXX", then raise an error and exit
if [ "$HF_TOKEN" == "XXX" ]; then
    echo "HF_TOKEN is not set, please set it in this script (some inference scripts need to access Hugging Face)"
    exit 1
fi