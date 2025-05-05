#!/bin/bash
sudo ifconfig eth0 down
sudo ifconfig eth0 mtu 9000
sudo ifconfig eth0 up

