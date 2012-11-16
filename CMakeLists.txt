CMAKE_MINIMUM_REQUIRED (VERSION 2.6) 
PROJECT(liver-surgery)

# Project is written in python, CMake is used for multiplatform deployment


ADD_CUSTOM_TARGET(git_recursive_pull ALL
    COMMAND git submodule update --init --recursive

    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )



FILE(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/sample_data )
FILE(DOWNLOAD "http://www.mathworks.com/matlabcentral/fileexchange/2762-dicom-example-files?download=true"
    "${CMAKE_SOURCE_DIR}/sample_data/head.zip"
    )
#ADD_CUSTOM_COMMAND(
#    OUTPUT sample_brain_unzip
#    COMMAND "unzip ${CMAKE_SOURCE_DIR} "
#    
#    
#    )

ADD_CUSTOM_TARGET(sample_brain ALL
    COMMAND unzip ${CMAKE_SOURCE_DIR}/sample_data/head.zip 
    COMMAND echo "ahoj"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/sample_data/
    )




#FILE ( REMOVE "${CMAKE_SOURCE_DIR}/sample_data/head.zip" )

    #DEPENDS sample_brain_unzip
    #    COMMAND
    #)








    #file(DOWNLOAD "file:///home/mjirik/projects/jatra/TODO.txt" "/home/mjirik/todo.txt") 


    # ----------------- pokus o crossplatform unzip     
    #    FIND_PROGRAM(UNZIP_EXECUTABLE wzzip PATHS "$ENV{ProgramFiles}/WinZip")
    #    IF(UNZIP_EXECUTABLE)
    #        SET(UNZIP_COMMAND "\"${UNZIP_EXECUTABLE}\" -P \"<ARCHIVE>\" @<FILELIST>")
    #ENDIF(ZIP_EXECUTABLE)
    #
    #IF (NOT ZIP_EXECUTABLE)
    #    FIND_PROGRAM(ZIP_EXECUTABLE 7z PATHS "$ENV{ProgramFiles}/7-Zip") 
    #    IF(ZIP_EXECUTABLE)
    #        SET(ZIP_COMMAND "\"${UNZIP_EXECUTABLE}\" a -tzip \"<ARCHIVE>\" @<FILELIST>")
    #    ENDIF(ZIP_EXECUTABLE)
    #ENDIF(NOT ZIP_EXECUTABLE)
    #
# --------- pokus o stazeni dat primo

#set(EXAMPLE_BRAIN_URL "http://www.mirthcorp.com/community/fisheye/chart/Mirth/branches/1.7.0/server/src/com/webreach/mirth/model/dicom/samplemessages")
#
#
#
#
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_001.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_001.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_002.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_002.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_003.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_003.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_004.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_004.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_005.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_005.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_006.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_006.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_007.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_007.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_008.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_008.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_009.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_009.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_010.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_010.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_011.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_011.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_012.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_012.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_013.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_013.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_014.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_014.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_015.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_015.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_016.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_016.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_017.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_017.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_018.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_018.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_019.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_019.dcm")
#file ( DOWNLOAD "${EXAMPLE_BRAIN_URL}/brain_020.dcm" "${CMAKE_SOURCE_DIR}/sample_data/brain_020.dcm")
