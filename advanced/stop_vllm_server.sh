#!/bin/bash
pkill -f "vllm serve.*--port $PORT"
pkill -f "vllm*"
