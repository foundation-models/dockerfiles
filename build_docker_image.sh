
# export docker_repo=docker.artifactory.dev.intapp.com/ai


for TAG in base api ml
do
    docker build -f Dockerfile.python3.9-cuda12-runtime-$TAG -t python3.9-cuda12-runtime:$TAG .
    docker tag python3.9-cuda12-runtime:$TAG "$docker_repo"/python3.9-cuda12-runtime:$TAG
    docker push "$docker_repo"/python3.9-cuda12-runtime:$TAG
done