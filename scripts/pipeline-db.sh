#! /usr/bin
# kgsf
model="kgsf"

# tgredial cn-dbpedia: 0.5, 0.7, 0.9, 1, 3
nhop="0.5"
nkey="0.5"
. scripts/tgredial-cndbpedia.sh

nhop="0.7"
nkey="0.7"
. scripts/tgredial-cndbpedia.sh

nhop="0.9"
nkey="0.9"
. scripts/tgredial-cndbpedia.sh

nhop="1"
nkey="one"
. scripts/tgredial-cndbpedia.sh

nhop="3"
nkey="three"
. scripts/tgredial-cndbpedia.sh

# inspired dbpedia: 0.5, 0.7, 0.9, 1, 2, 3
nhop="0.5"
nkey="0.5"
. scripts/inspired-dbpedia.sh

nhop="0.7"
nkey="0.7"
. scripts/inspired-dbpedia.sh

nhop="0.9"
nkey="0.9"
. scripts/inspired-dbpedia.sh

nhop="1"
nkey="one"
. scripts/inspired-dbpedia.sh

nhop="2"
nkey="two"
. scripts/inspired-dbpedia.sh

nhop="3"
nkey="three"
. scripts/inspired-dbpedia.sh
