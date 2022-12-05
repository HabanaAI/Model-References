#!/bin/bash

#Description
#This script outputs a file for each moduleID.
#These files contain the Hthread_sequence on which the process is bound too (this is a restriction and not a reservation).
#We do this by getting the mapping of (ModuleID, pcie_bus_id) from hl-smi
#Then we map the 2 tuple to a numa by opening the file
#/sys/bus/pci/devices/<pcie_bus_id >/numa_node
#Now we have a 3 tuple (ModuleID, pcie_bus_id,  numa_node)
#Lastly we get the Hthread_sequence that correspond to that numa_node from lscpu so now we have
#(ModuleID, pcie_bus_id,  numa_node, Hthread_sequence )
#The Hthread_sequence is then used to bind the process to the specific threads on the numa closest to the PCIE bus.

affinity_print()
{
   echo "Affinity: "$1
}

hl_smi_check()
{
   if [[ ! -x `which hl-smi` ]];
   then
         affinity_print "hl-smi could not be found, exiting"
         exit 1
   fi
}

check_env()
{
   if [[ -z "$NUMA_MAPPING_DIR" ]];
   then
         affinity_print "Missing env variable \"NUMA_MAPPING_DIR\", exiting!"
         exit 1
   fi
}

create_temp_files()
{
   # create a temp directory, mktemp is used to create a temp directory with a unique name
   temp_dir=$(mktemp -d)

   # create temp files for holding outputs
   file_hl_smi=$temp_dir/hl_smi.txt
   file_module_id=$temp_dir/module_id.txt
   file_pcie_bus_id=$temp_dir/pcie_bus_id.txt
   file_pcie_numa=$temp_dir/pcie_numa.txt
   file_hl_smi=$temp_dir/hl_smi.txt
   file_configuration_table=$temp_dir/configuration_table.txt
   file_final_output=$NUMA_MAPPING_DIR/.habana_module_topo
}

create_configuartion_table()
{
   # save the entire hl-smi output to file
   hl-smi -L > $file_hl_smi

   #check that the driver is up
   if [ $? -eq 1 ]; then
      affinity_print "Issue while trying to run hl-smi, aborting..."
      exit 1
   fi

   # get the module IDs (unique identifier for each gaudi)
   grep "Module ID" $file_hl_smi > $file_module_id

   # get the bus IDs
   grep "Bus Id" $file_hl_smi > $file_pcie_bus_id

   # Get the numa for each PCIE bus
   for i in `cat $file_pcie_bus_id|awk '{print $4}'`; do
      numa_node=`cat /sys/bus/pci/devices/$i/numa_node`
      if [ $numa_node -ge 0 ]; then
         echo $numa_node >> $file_pcie_numa
      else
         for i in `hl-smi -L|grep "Bus Id"|awk '{print $4}'`; do affinity_print "PCIE:"$i", NUMA:"`cat /sys/bus/pci/devices/$i/numa_node`; done
         affinity_print "Numa mapping isn't set properly, you are most likley running on an unsupported VM, aborting..."
         exit 1
      fi
   done

   #append output files horizontally
   paste $file_module_id $file_pcie_bus_id $file_pcie_numa | awk ' {print $4,$8,$9}' | sort -k1 > $file_configuration_table
}


create_thread_list()
{
   no_of_numa_nodes=`lscpu|grep "NUMA node(s):"|awk '{print $3}'`
   no_of_gaudis=`cat $file_configuration_table|wc -l`
   no_of_used_numa=`cat $file_pcie_numa | uniq | wc -l`


   for module_id in $(seq 0 $(($no_of_gaudis-1))); do
      #grab one pcieid at a time (busID)
      pcie_bus_id=`cat $file_configuration_table | awk '{print $2}' | sed -n $(($module_id+1))p`

      #get the corespoinding numanode (pcie_numa)
      numa_node=`cat /sys/bus/pci/devices/$pcie_bus_id/numa_node`

      #special barcelona configuration where two sockets are configured to be 4 virtual numa nodes
      if [[ $no_of_used_numa -eq 2 && $no_of_numa_nodes -eq 4 ]]; then
         #get current node (moduleID)
         curr_node=`cat $file_configuration_table | awk '{print ","$3,$1}'| grep ",$numa_node" | awk '{print $2}'|head -1`
         if [ $module_id -eq $curr_node ]; then
            numa_node=$(($numa_node-1))
         fi
      fi

      #get the list of threads
      if [ $numa_node -ge 0 ]; then
         vector=`lscpu --parse | grep ",$numa_node,,"|awk -F"," '{print $1}'`
         vector=`echo $vector | tr ' ' ,`
         echo $vector > $NUMA_MAPPING_DIR/.habana_moduleID$module_id
         echo $vector >> $temp_dir/.module
      fi
   done
}


add_thread_list_to_config_table()
{
   #put it all together
   echo "ModID   BusID  NUMA   CPUs: " > $file_final_output
   echo "=====   =====  =====  ===== " >> $file_final_output
   paste $file_configuration_table $temp_dir/.module >> $file_final_output
}

clean_up()
{
   #remove the temp dir
   if [ ! -z "$temp_dir" ]; then
      rm -fr $temp_dir
   fi
}

main()
{
   check_env
   hl_smi_check
   create_temp_files
   create_configuartion_table
   create_thread_list
   add_thread_list_to_config_table
   clean_up
   affinity_print "Script finished successfully"
   exit 0
}

main