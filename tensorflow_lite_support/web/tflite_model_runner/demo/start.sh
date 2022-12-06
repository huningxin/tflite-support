#!/bin/bash

cp -f src/* dist/
npx http-server dist/ -S
