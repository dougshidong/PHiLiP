#!/bin/bash 

#VARIABLE_TO_REPLACE="surface\ volume_nodes"
#NEW_VARIABLE_NAME="surface\ nodes"
VARIABLE_TO_REPLACE="RadFadtype"
NEW_VARIABLE_NAME="RadFadType"

#VARIABLE_TO_REPLACE="surface_to_volume_indices"
#NEW_VARIABLE_NAME="surface_indices"

SED_CMD="xargs sed -i 's/\b'${VARIABLE_TO_REPLACE}'\b/'${NEW_VARIABLE_NAME}'/g'"
#SED_CMD="xargs sed -i 's/ \+\n/\r/g'"
#SED_CMD="xargs sed -i 's/'${VARIABLE_TO_REPLACE}'/'${NEW_VARIABLE_NAME}'/g'"

echo $SED_CMD
grep -rl "${VARIABLE_TO_REPLACE}" \
    "src" \
    | eval ${SED_CMD}

grep -rl "${VARIABLE_TO_REPLACE}" \
    "tests" \
    | eval ${SED_CMD}
