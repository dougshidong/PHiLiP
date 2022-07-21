configure_file(naca0012_hopw_ref2.msh naca0012.msh COPYONLY)
configure_file(naca0012_hopw_ref0.msh naca0012_hopw_ref0.msh COPYONLY)
configure_file(naca0012_hopw_ref1.msh naca0012_hopw_ref1.msh COPYONLY)
configure_file(naca0012_hopw_ref2.msh naca0012_hopw_ref2.msh COPYONLY)
configure_file(naca0012_hopw_ref3.msh naca0012_hopw_ref3.msh COPYONLY)
configure_file(naca0012_hopw_ref4.msh naca0012_hopw_ref4.msh COPYONLY)
configure_file(naca0012_hopw_ref5.msh naca0012_hopw_ref5.msh COPYONLY)

set (filename "naca0012_hopw_ref0.msh")
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
    message(SEND_ERROR
            "Missing NACA0012 files named ${filename}. Please download them from
    https://drive.google.com/drive/folders/182JusbWV6NAA8ws1-TTg7M2GLc5jt6_r?usp=sharing
and place them in /tests/meshes."
            )
endif()

set (filename "naca0012_hopw_ref1.msh")
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
    message(SEND_ERROR
            "Missing NACA0012 files named ${filename}. Please download them from
    https://drive.google.com/drive/folders/182JusbWV6NAA8ws1-TTg7M2GLc5jt6_r?usp=sharing
and place them in /tests/meshes."
            )
endif()

set (filename "naca0012_hopw_ref2.msh")
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
    message(SEND_ERROR
            "Missing NACA0012 files named ${filename}. Please download them from
    https://drive.google.com/drive/folders/182JusbWV6NAA8ws1-TTg7M2GLc5jt6_r?usp=sharing
and place them in /tests/meshes."
            )
endif()