#!/bin/bash

echo "You will be deleting all the following vim temporary files: " 
echo    # (optional) move to a new line
ls ./**/.*.swp
echo    # (optional) move to a new line
read -p "Are you sure? (y/n)" answer
echo    # (optional) move to a new line
if [[ $answer =~ ^[Yy]$ ]]; then
    rm ./**/.*.swp
else
    echo "Cancelling deletion.."
fi
