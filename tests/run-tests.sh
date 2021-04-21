
# verbose mode, exit on error
set -ex

# parse command line arguments
# while getopts ":u" flag
# do
#     case "${flag}" in
#        u) upload_codecov=true;;
#     esac
# done

# if [ "$upload_codecov" = true ] ; then
#     echo 'Code coverage upload flag set. Will upload report to codecov.io'
# fi

pip3 install -U pytest # unit test tool
pip3 install -U codecov # code coverage tool
pip3 install -U pytest-cov # coverage plugin for pytest
pip3 install -U pylint # python linter
pip3 install -U dynaconf
BASEDIR=$(dirname $0)
pip3 install -e $BASEDIR/../src
echo "Script location: ${BASEDIR}"
# change work dir location to a predictable place
# where codecov can find the generated reports
cd $BASEDIR/../
echo PWD=$PWD
python3 -m pytest --cov=ambianic --cov-report=xml --cov-report=term tests/

# if -u command line argument is passed, submit code coverage report to codecov.io
# parse command line arguments
# if [ "$upload_codecov" = true ] ; then
#   echo 'Uploading code coverage report to codecov.io'
#    codecov
# fi

