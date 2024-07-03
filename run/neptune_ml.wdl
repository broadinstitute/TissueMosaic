version 1.0

task train {
    input {
        File MAIN_PY
        File ML_CONFIG
        File anndata_tar_gz
        File ckpt_previous_run
        File credentials_json
        String git_repo
        String git_branch_or_commit
        Int cpus_count
        Int gpus_count
        String gpus_type
    }


    command <<<

        exec_dir=$(pwd)
        echo "--> $exec_dir"
        echo "START --> Content of exectution dir"
        echo $(ls)

        # 2. clone the repository in the checkout_dir
        
        # for public repository use:
        # git clone ~{git_repo} ./checkout_dir
        
        # for private repository use:
        github_token=$(cat ~{credentials_json} | grep -o '"GITHUB_API_TOKEN"\s*:\s*"[^"]*"' | grep -o '"[^"]*"$' | sed 's/"//g')
        git_repo_with_token=$(echo ~{git_repo} | sed "s/github/$github_token@github/")
        git clone $git_repo_with_token ./checkout_dir

        # 3. checkout the branch
        cd ./checkout_dir
        git checkout ~{git_branch_or_commit}
        
        # 4. Install the package
        #    and create links from delocalized files and give them the name you expects
        pip install .  # this means that your package has a setup.py file

        # 5. prepare the files
        mkdir -p ./data_folder
        tar -xzf ~{anndata_tar_gz} -C ./data_folder
        ln -s ~{MAIN_PY} ./main.py
        ln -s ~{ML_CONFIG} ./config.yaml
        ln -s $exec_dir/my_checkpoint.ckpt ./preemption_ckpt.pt  # this is to resume a pre-empted run     (it has precedence)
        ln -s ~{ckpt_previous_run} ./old_run_ckpt.pt             # this is to resume from a previous run  (secondary)

        # Install missing packages not already included in the docker image (if any)
        # pip install xxxx

        # 5. run python code only if NEPTUNE credentials are found
        neptune_token=$(cat ~{credentials_json} | grep -o '"NEPTUNE_API_TOKEN"\s*:\s*"[^"]*"' | grep -o '"[^"]*"$')
        if [ ! -z $neptune_token ]; then
           export NEPTUNE_API_TOKEN=$neptune_token
           python main.py --config ./config.yaml --data_folder data_folder --gpus ~{gpus_count}
        fi
    >>>
    
    runtime {
         docker: "us.gcr.io/broad-dsde-methods/tissuemosaic:0.0.4"
         bootDiskSizeGb: 200
         memory: "26G"
         cpu: cpus_count
         zones: "us-east1-d us-east1-c"
         gpuCount: gpus_count
         gpuType: gpus_type
         maxRetries: 1
         preemptible: 3
         checkpointFile: "my_checkpoint.ckpt"  # don't change this name. The code above and the logger in pytorch-lightining rely on it
    }

}

workflow neptune_ml {

    input {
        File MAIN_PY
        File ML_CONFIG
        File anndata_tar_gz
        File ckpt_previous_run
        File credentials_json
        String git_repo
        String git_branch_or_commit 
        Int cpus_count
        Int gpus_count
        String gpus_type
    }

    call train { 
        input :
            MAIN_PY = MAIN_PY,
            ML_CONFIG = ML_CONFIG,
            anndata_tar_gz = anndata_tar_gz,
            credentials_json = credentials_json,
            ckpt_previous_run = ckpt_previous_run,
            git_repo = git_repo,
            git_branch_or_commit = git_branch_or_commit,
            cpus_count = cpus_count,
            gpus_count = gpus_count,
            gpus_type = gpus_type
    }
}
