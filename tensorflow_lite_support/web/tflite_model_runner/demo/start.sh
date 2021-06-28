#!/bin/bash

cp -f src/* dist/
cp -r deps/webnn-polyfill dist/
npx http-server dist/
