git submodule init
git submodule update
git config --global http.proxy ""
git pull --recurse-submodules
git submodule update --recursive