#! /usr/bin
# kgsf
model="kgsf"

# inspired conceptnet: 0.5, 0.7, 0.9, 1, 2, 3
nhop="0.5"
nkey="0.5"
. scripts/inspired-conceptnet.sh

nhop="0.7"
nkey="0.7"
. scripts/inspired-conceptnet.sh

nhop="0.9"
nkey="0.9"
. scripts/inspired-conceptnet.sh

nhop="1"
nkey="one"
. scripts/inspired-conceptnet.sh

nhop="2"
nkey="two"
. scripts/inspired-conceptnet.sh

nhop="3"
nkey="three"
. scripts/inspired-conceptnet.sh

# tgredial hownet: 0.5, 0.7, 0.9, 1, 2, 3
nhop="0.5"
nkey="0.5"
. scripts/tgredial-hownet.sh

nhop="0.7"
nkey="0.7"
. scripts/tgredial-hownet.sh

nhop="0.9"
nkey="0.9"
. scripts/tgredial-hownet.sh

nhop="1"
nkey="one"
. scripts/tgredial-hownet.sh

nhop="2"
nkey="two"
. scripts/tgredial-hownet.sh

nhop="3"
nkey="three"
. scripts/tgredial-hownet.sh
