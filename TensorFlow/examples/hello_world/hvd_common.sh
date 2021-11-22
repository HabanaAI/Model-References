export CARDS_PER_NODE=${CARDS_PER_NODE:-$((`hl-smi -Q bus_id --format=csv | wc -l` - 1))}

gen_host_ip_list() {
    if [ -z $CARDS_PER_NODE ]; then
        echo "variable CARDS_PER_NODE is not specified!!"
        exit 1
    fi
    HOST_LIST=
    for ip in $@; do
        for _ in `seq $CARDS_PER_NODE`; do
            if [ -z $HOST_LIST ]; then
                HOST_LIST=${ip}
            else
                HOST_LIST=${HOST_LIST},${ip}
            fi
        done
    done
    export HOST_LIST=$HOST_LIST
}

gen_if_include_list() {
    MPI_TCP_INCLUDE=
    # Below command removes duplicated Host addresses to avoid defining
    # redundant subnets.
    for ip in `echo $1 | tr ',' '\n' | sort | uniq | xargs`; do
        if [ -z $MPI_TCP_INCLUDE ]; then
            MPI_TCP_INCLUDE=${ip}/24
        else
            MPI_TCP_INCLUDE=${MPI_TCP_INCLUDE},${ip}/24
        fi
    done
    export MPI_TCP_INCLUDE=$MPI_TCP_INCLUDE
}
