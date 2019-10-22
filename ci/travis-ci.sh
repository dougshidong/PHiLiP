#
# @authors Andrei Novikov (pyclustering@yandex.ru)
# @date 2014-2019
# @copyright GNU Public License
#
# GNU_PUBLIC_LICENSE
#   pyclustering is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   pyclustering is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


# CCORE_X64_BINARY_FOLDER=pyclustering/core/x64/linux
# CCORE_X64_BINARY_PATH=$CCORE_X64_BINARY_FOLDER/ccore.so
# 
# CCORE_X86_BINARY_FOLDER=pyclustering/core/x86/linux
# CCORE_X86_BINARY_PATH=$CCORE_X86_BINARY_FOLDER/ccore.so
# 
DOXYGEN_FILTER=( "warning: Unexpected new line character" )


print_error() {
    echo "[PHiLiP CI] ERROR: $1"
}


print_info() {
    echo "[PHiLiP CI] INFO: $1"
}


check_failure() {
    if [ $? -ne 0 ] ; then
        if [ -z $1 ] ; then
            print_error $1
        else
            print_error "Failure exit code is detected."
        fi
        exit 1
    fi
}


check_error_log_file() {
    problems_amount=$(cat $1 | wc -l)
    printf "Total amount of errors and warnings: '%d'\n"  "$problems_amount"
    
    if [ $problems_amount -ne 0 ] ; then
        print_info "List of warnings and errors:"
        cat $1
        
        print_error $2
        exit 1
    fi
}


build_ccore() {
    cd $TRAVIS_BUILD_DIR/ccore/

    [ -f stderr.log ] && rm stderr.log
    [ -f stdout.log ] && rm stdout.log
    
    if [ "$1" == "x64" ]; then
        make ccore_x64 > >(tee -a stdout.log) 2> >(tee -a stderr.log >&2)
        check_error_log_file stderr.log "Building CCORE (x64): FAILURE."
    elif [ "$1" == "x86" ]; then
        make ccore_x86 > >(tee -a stdout.log) 2> >(tee -a stderr.log >&2)
        check_error_log_file stderr.log "Building CCORE (x86): FAILURE."
    else
        print_error "Unknown CCORE platform is specified."
        exit 1
    fi

    cd -
}


run_build_ccore_job() {
    print_info "CCORE (C++ code building):"
    print_info "- Build CCORE library for x64 platform."
    print_info "- Build CCORE library for x86 platform."

    #install requirement for the job
    print_info "Install requirement for CCORE building."

    sudo apt-get install -qq g++-5
    sudo apt-get install -qq g++-5-multilib
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50

    # show info
    g++ --version
    gcc --version

    # build ccore library
    build_ccore x64
    build_ccore x86

    print_info "Upload ccore x64 binary."
    upload_binary x64 linux
    
    print_info "Upload ccore x86 binary."
    upload_binary x86 linux
}

run_valgrind_ccore_job() {
    print_info "VALGRIND CCORE (C++ code valgrind shock checking):"
    print_info "- Run unit-tests of pyclustering."
    print_info "- Shock memory leakage detection by valgrind."

    # install requirements for the job
    sudo apt-get install -qq g++-5
    sudo apt-get install -qq g++-5-multilib
    sudo apt-get install -qq valgrind
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50

    # build and run unit-test project under valgrind to check memory leakage
    cd ccore/

    make valgrind_shock
    check_failure "CCORE shock memory leakage status: FAILURE."
}

run_integration_test_job() {
    print_info "INTEGRATION TESTING ('ccore' <-> 'pyclustering' for platform '$1')."
    print_info "- Build CCORE library."
    print_info "- Run integration tests of pyclustering."

    PLATFORM_TARGET=$1

    # install requirements for the job
    install_miniconda $PLATFORM_TARGET

    sudo apt-get install -qq g++-5 gcc-5
    sudo apt-get install -qq g++-5-multilib gcc-5-multilib
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50
    sudo update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-5 50

    # build ccore library
    build_ccore $PLATFORM_TARGET

    # run integration tests
    python pyclustering/tests/tests_runner.py --integration
}


run_doxygen_job() {
    print_info "DOXYGEN (documentation generation)."
    print_info "- Generate documentation and check for warnings."

    sudo apt update
    print_info "Install cmake"
    sudo apt install cmake

    print_info "Install doxygen"
    sudo apt install doxygen

    print_info "Install requirements for doxygen."
    sudo apt install graphviz
    sudo apt install texlive

    rm -rf build_doc;
    mkdir build_doc; cd build_doc;
    cmake ../ -DDOC_ONLY=ON

    print_info "Prepare log files."
    report_file=doxygen_problems.log
    report_file_filtered=doxygen_problems_filtered.log

    rm -f $report_file
    rm -f $report_file_filtered

    print_info "Generate documentation."
    doxygen --version
#make doc 2>&1 | tee $report_file > >(while read line; do echo -e "\e[01;31m$line\e[0m" >&2; done)
    make doc > $report_file 2>&1
    awk '/error|warning/ {print}' $report_file > $report_file_filtered
    cat $report_file_filtered

    check_error_log_file $report_file_filtered "Building doxygen documentation: FAILURE."
    print_info "Finished doxygen documentation test. SUCCESS."
}


set -e
set -x

if [[ $TRAVIS_COMMIT_MESSAGE == *"[no-build]"* ]]; then
    print_info "Option '[no-build]' is detected, sources will not be built, checked, verified and published."
    exit 0
fi

if [[ $TRAVIS_COMMIT_MESSAGE == *"[build-only-osx]"* ]]; then
    if [[ $1 == BUILD_TEST_CCORE_MACOS ]]; then
        print_info "Option '[build-only-osx]' is detected, mac os build will be started."
    else
        print_info "Option '[build-only-osx]' is detected, sources will not be built, checked, verified and published."
        exit 0
    fi
fi

case $1 in
    BUILD_CCORE) 
        run_build_ccore_job ;;

    ANALYSE_CCORE)
        run_analyse_ccore_job ;;

    UT_CCORE) 
        run_ut_ccore_job ;;

    VALGRIND_CCORE)
        run_valgrind_ccore_job ;;

    TEST_PYCLUSTERING) 
        run_test_pyclustering_job ;;

    IT_CCORE_X86)
        run_integration_test_job x86 ;;

    IT_CCORE_X64)
        run_integration_test_job x64 ;;

    BUILD_TEST_CCORE_MACOS)
        run_build_test_ccore_macos_job ;;

    DOCUMENTATION)
        run_doxygen_job ;;

    DEPLOY)
        run_deploy_job ;;

    *)
        print_error "Unknown target is specified: '$1'"
        exit 1 ;;
esac
