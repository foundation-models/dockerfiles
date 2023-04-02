
# export docker_repo=hossein20s/ai


for TAG in base api ml non-root-ml model cmd
do
    # docker pull "$docker_repo"/python3.9-cuda12-runtime:$TAG
    # docker tag "$docker_repo"/python3.9-cuda12-runtime:$TAG python3.9-cuda12-runtime:$TAG
    docker build -f Dockerfile.python3.9-cuda12-runtime-$TAG -t python3.9-cuda12-runtime:$TAG .
    docker tag python3.9-cuda12-runtime:$TAG "$docker_repo"/python3.9-cuda12-runtime:$TAG
    docker push "$docker_repo"/python3.9-cuda12-runtime:$TAG
done