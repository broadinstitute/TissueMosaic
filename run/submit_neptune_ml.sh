#!/usr/bin/env bash

# Define some defaults
DEFAULT_BUCKET="gs://ld-tmp-storage/input_jsons"
DEFAULT_WDL="neptune_ml.wdl"
DEFAULT_WDL_JSON="WDL_parameters.json"
DEFAULT_ML_CONFIG="ML_parameters.yaml"
DEFAULT_MAIN_PY="main.py"

# Set variables to default
BUCKET=$DEFAULT_BUCKET
WDL=$DEFAULT_WDL
WDL_JSON=$DEFAULT_WDL_JSON
ML_CONFIG=$DEFAULT_ML_CONFIG
MAIN_PY=$DEFAULT_MAIN_PY
HERE=${PWD}
SCRIPTNAME=$( echo $0 | sed 's#.*/##g' )

# Helper functions
display_help() {
  echo -e ""
  echo -e "-- $SCRIPTNAME --"
  echo -e ""
  echo -e " Submit wdl workflow using cromshell."
  echo -e ""
  echo -e " Example usage:"
  echo -e "   $SCRIPTNAME $WDL --py $MAIN_PY --wdl $WDL_JSON --ml $ML_CONFIG -b $BUCKET"
  echo -e "   $SCRIPTNAME -h"
  echo -e ""
  echo -e " Supported Flags:"
  echo -e "   -h or --help     Display this message"
  echo -e "   -p or --py       Name of file python file to run, usually main.py. This file will be copied as-is to the VM"
  echo -e "   -m or --ml       Name of file with all the parameters for the ML model. This file will be copied as-is to the VM"
  echo -e "   -w or --wdl      Name of json file with all the parameters for the WDL." 
  echo -e "   -b or --bucket   Name of google bucket where local files will be copied (VM will then localize those files)"
  echo -e "   -t or --template Show the template for $WDL_JSON" 
  echo -e ""
  echo -e " Default behavior (can be changed manually by editing the $SCRIPTNAME):"
  echo -e "   If no inputs are specified the default values will be used:"
  echo -e "   main_file ----------> $MAIN_PY"
  echo -e "   wdl_file -----------> $WDL"
  echo -e "   wdl_json_file ------> $WDL_JSON"
  echo -e "   ml_config_file -------> $ML_CONFIG"
  echo -e "   bucket -------------> $BUCKET"
  echo -e ""
  echo -e ""
}


exit_with_error() {
  echo -e ""
  echo -e "ERROR!. Something went wrong"
  exit 1
}

exit_with_success() {
  echo -e ""
  echo -e "GREAT!. Everything went smoothly"
  exit 0
}

template_wdl_json() {
  echo -e ""
  echo -e "Based on $WDL the template for $WDL_JSON is:"
  womtool inputs $WDL | sed '/ML_parameters/d'
}

#--------------------------------------
# 1. read inputs from command line
#--------------------------------------

while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help)
			display_help
			exit 0
			;;
		-w|--wdl)
			WDL_JSON=$2
			shift 2
			;;
		-p|--py)
			MAIN_PY=$2
			shift 2
			;;
		-m|--ml)
			ML_CONFIG=$2
			shift 2
			;;
		-b|--bucket)
			BUCKET=$2
			shift 2
			;;
		-t|--template)
			template_wdl_json
			exit 0
			;;
		-*|--*) # unknown option
			echo "ERROR: Unsupported flag $1"
			exit 1
			;;
		*.wdl) # wdl workflow
			WDL=$1
			shift 1
			;;
		*)
			echo "ERROR: Unrecognized option $1"
			exit 1
			;;
	esac
done  # end of while loop

# At this point I have these trhee values:
echo "Current values: -->" $WDL $MAIN_PY $WDL_JSON $ML_CONFIG $BUCKET

# 1. copy ML_CONFIG and MAIN_PY in the cloud with random hash
echo
echo "Step1: copying $ML_CONFIG  into google bucket"
RANDOM_HASH=$(cat /dev/urandom | od -vAn -N8 -tx8 | head -1 | awk '{print $1}')
ML_CONFIG_CLOUD="$BUCKET/${RANDOM_HASH}_$ML_CONFIG"
MAIN_PY_CLOUD="$BUCKET/${RANDOM_HASH}_$MAIN_PY"
gsutil cp $ML_CONFIG $ML_CONFIG_CLOUD 
if [ "$?" != "0" ]; then
   exit_with_error
else
   echo "copied $ML_CONFIG to $ML_CONFIG_CLOUD"
fi
gsutil cp $MAIN_PY $MAIN_PY_CLOUD 
if [ "$?" != "0" ]; then
   exit_with_error
else
   echo "copied $MAIN_PY to $MAIN_PY_CLOUD"
fi


# 2. create the json file which will be passed to cromshell
echo
echo "Step2: crerating input.json file for cromshell"
womtool inputs $WDL | jq 'keys[]' > tmp_list.txt
key_for_ML_parameters=$(cat tmp_list.txt | grep "ML_CONFIG")
key_for_MAIN_PY=$(cat tmp_list.txt | grep "MAIN_PY")
rm -rf tmp_list.txt

# 2.1 Make a simple json with the path to MAIN_PY_CLOUD and ML_CONFIG_CLOUD
echo '{' > tmp.json
echo "$key_for_ML_parameters" : '"'"$ML_CONFIG_CLOUD"'",' >> tmp.json
echo "$key_for_MAIN_PY" : '"'"$MAIN_PY_CLOUD"'"' >> tmp.json
echo '}' >> tmp.json

# 2.2 Merge the two json 
jq -s '.[0] * .[1]' tmp.json $WDL_JSON | tee input.json
if [ "$?" != "0" ]; then
   exit_with_error
fi
rm -rf tmp.json

# 3. run cromshell
echo
echo "Step3: run cromshell"
echo "RUNNING: cromshell submit $WDL input.json"
cromshell submit $WDL input.json | tee tmp_run_status
if [ "$?" != "0" ]; then
   exit_with_error
fi
rm -rf input.json 

# 4. check I find the word Submitted in the cromshell_output
echo
echo "Step4: check submission"
read -r ID_1 STATUS <<<$(tail -1 tmp_run_status | jq '.id, .status' | sed 's/\"//g')
rm tmp_run_status
if [ "$STATUS" != "Submitted" ]; then
   exit_with_error
fi

exit_with_success
