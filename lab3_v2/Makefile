SHELL := /bin/bash

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: setup

setup:
	git clone https://github.com/coqui-ai/TTS
	cd TTS
	sudo apt update
	sudo apt install python3-dev build-essential
	sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev
	pip install -U pip
	pip install -e .
