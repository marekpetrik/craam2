#!/bin/sh

rsync -rvP domains marek@rmdp.xyz:/srv/data
rsync -vP README.md marek@rmdp.xyz:/srv/data/domains
