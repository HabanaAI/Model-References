export CARDS_PER_NODE=${CARDS_PER_NODE:-8}

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

gen_multinode_hclconfig() {
    if [ -z $CARDS_PER_NODE ]; then
        echo "variable CARDS_PER_NODE is not specified!!"
        exit 1
    fi

    if [ ${#@} -gt 1 ] && [ -z $HOST_LIST ] ; then
        echo "variable HOST_LIST must be specified for multiple nodes!!"
        exit 1
    fi

    export HCL_CONFIG_PATH=/tmp/hcl_config.`date +%s`.json
    echo "Creating config file at "${HCL_CONFIG_PATH}
    touch ${HCL_CONFIG_PATH}

    echo "{"                                            >> ${HCL_CONFIG_PATH}
    echo "    \"HCL_PORT\": 5332,"                      >> ${HCL_CONFIG_PATH}
    echo "    \"HCL_TYPE\": \"HLS1\","                  >> ${HCL_CONFIG_PATH}
    if [ ${#@} -lt 2 ]; then
        echo "    \"HCL_COUNT\": $CARDS_PER_NODE"       >> ${HCL_CONFIG_PATH}
    else
        echo "    \"HCL_RANKS\": ["                     >> ${HCL_CONFIG_PATH}
        LIST=`echo ${HOST_LIST} | sed 's#,#","#g'`
        echo "    \"${LIST}\""                          >> ${HCL_CONFIG_PATH}
        echo "    ]"                                    >> ${HCL_CONFIG_PATH}
    fi
    echo "}"                                            >> ${HCL_CONFIG_PATH}

    if [ ${#@} -gt 1 ]; then
        for ip in $@; do
            scp ${HCL_CONFIG_PATH} ${USER}@${ip}:${HCL_CONFIG_PATH}
        done
    fi
}
