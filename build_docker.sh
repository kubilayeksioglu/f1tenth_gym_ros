#!/bin/bash
if [ ! -d f1tenth_gym ] ; then
    git clone -b cpp_backend_archive --depth=1 https://github.com/f1tenth/f1tenth_gym
    cd f1tenth_gym
    git checkout 
    cd ../
else
    echo f1tenth_gym exists, not cloning.
fi
docker build -t f1tenth_gym -f Dockerfile .