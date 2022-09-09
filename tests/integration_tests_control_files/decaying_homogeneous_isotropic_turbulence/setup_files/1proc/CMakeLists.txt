configure_file(setup_philip-00000.dat setup_philip-00000.dat COPYONLY)

set (filename "setup_philip-00000.dat")
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
    message(SEND_ERROR
            "Missing flow initialization files named ${filename}.
             Please run PHiLiP/get_flow_initialization_files_local.sh or 
             PHiLiP/get_flow_initialization_files_cluster.sh if compiling on the cluster."
            )
endif()