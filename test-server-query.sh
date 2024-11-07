#!/bin/bash
#
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"text": "author", "top_k": 5}'
