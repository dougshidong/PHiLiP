set(SMALL_TEST testrun)
set(TARGET ${SMALL_TEST})
set(SMALL_TEST_SRC 
    temporary_test.cpp)
add_executable(${SMALL_TEST} ${SMALL_TEST_SRC})
target_link_libraries(${SMALL_TEST} mpi )
DEAL_II_SETUP_TARGET(${SMALL_TEST})
message("Adding executable " ${SMALL_TEST} " with files " ${SMALL_TEST_SRC} "\n")
unset(SMALL_TEST)
unset(TARGET ${SMALL_TEST})

#set(SMALL_TEST instantiate_vector_ad)
#set(TARGET ${SMALL_TEST})
#set(SMALL_TEST_SRC 
#    instantiate_vector_ad.cpp)
#add_executable(${SMALL_TEST} ${SMALL_TEST_SRC})
#target_link_libraries(${SMALL_TEST} mpi )
#DEAL_II_SETUP_TARGET(${SMALL_TEST})
#message("Adding executable " ${SMALL_TEST} " with files " ${SMALL_TEST_SRC} "\n")
#unset(SMALL_TEST)
#unset(TARGET ${SMALL_TEST})

set(SMALL_TEST manifoldrun)
set(TARGET ${SMALL_TEST})
set(SMALL_TEST_SRC 
    test_manifold.cpp)

add_executable(${SMALL_TEST} ${SMALL_TEST_SRC})
#    target_link_libraries(${SMALL_TEST} m mpi)

DEAL_II_SETUP_TARGET(${SMALL_TEST})
message("Adding executable " ${SMALL_TEST} " with files " ${SMALL_TEST_SRC} "\n")

unset(SMALL_TEST)
unset(TARGET ${SMALL_TEST})
