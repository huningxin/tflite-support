#!/bin/bash

cp -rf src/* dist/
npx http-server dist/ -S
