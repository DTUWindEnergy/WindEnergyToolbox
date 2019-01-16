# build image
docker build -t dtuwindenergy/wetb ./
# or pull from docker hub
  
# create container
docker create -it --name wetb dtuwindenergy/wetb

# start container
docker start wetb

# enter container with bash prompt
docker exec -it wetb bash

# push
docker push dtuwindenergy/wetb

#checkout single branch for testing:
git clone --single-branch --branch test_pypi https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox.git

#Copy local folder into docker container:
docker cp "C:\Sandbox\Git\WindEnergyToolbox\." wetb:/WETB/
