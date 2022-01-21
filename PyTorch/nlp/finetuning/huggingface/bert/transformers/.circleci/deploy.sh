cd docs

function deploy_doc(){
	echo "Creating doc at commit $1 and pushing to folder $2"
	git checkout $1
	pip install -U ..
	if [ ! -z "$2" ]
	then
		if [ "$2" == "master" ]; then
		    echo "Pushing master"
			make clean && make html && scp -r -oStrictHostKeyChecking=no _build/html/* $doc:$dir/$2/
			cp -r _build/html/_static .
		elif ssh -oStrictHostKeyChecking=no $doc "[ -d $dir/$2 ]"; then
			echo "Directory" $2 "already exists"
			scp -r -oStrictHostKeyChecking=no _static/* $doc:$dir/$2/_static/
		else
			echo "Pushing version" $2
			make clean && make html
			rm -rf _build/html/_static
			cp -r _static _build/html
			scp -r -oStrictHostKeyChecking=no _build/html $doc:$dir/$2
		fi
	else
		echo "Pushing stable"
		make clean && make html
		rm -rf _build/html/_static
		cp -r _static _build/html
		scp -r -oStrictHostKeyChecking=no _build/html/* $doc:$dir
	fi
}

# You can find the commit for each tag on https://github.com/huggingface/transformers/tags
deploy_doc "master" master
deploy_doc "b33a385" v1.0.0
deploy_doc "fe02e45" v1.1.0
deploy_doc "89fd345" v1.2.0
deploy_doc "fc9faa8" v2.0.0
deploy_doc "3ddce1d" v2.1.1
deploy_doc "3616209" v2.2.0
deploy_doc "d0f8b9a" v2.3.0
deploy_doc "6664ea9" v2.4.0
deploy_doc "fb560dc" v2.5.0
deploy_doc "b90745c" v2.5.1
deploy_doc "fbc5bf1" v2.6.0
deploy_doc "6f5a12a" v2.7.0
deploy_doc "11c3257" v2.8.0
deploy_doc "e7cfc1a" v2.9.0
deploy_doc "7cb203f" v2.9.1
deploy_doc "10d7239" v2.10.0
deploy_doc "b42586e" v2.11.0
deploy_doc "7fb8bdf" v3.0.2
deploy_doc "4b3ee9c" v3.1.0
deploy_doc "3ebb1b3" v3.2.0
deploy_doc "0613f05" v3.3.1
deploy_doc "eb0e0ce" v3.4.0
deploy_doc "818878d" v3.5.1
deploy_doc "c781171" v4.0.1
deploy_doc "bfa4ccf" v4.1.1
deploy_doc "7d9a9d0" v4.2.2
deploy_doc "bae0c79" v4.3.3
deploy_doc "c988db5" v4.4.0
deploy_doc "c5d6a28" v4.4.1
deploy_doc "6bc89ed" v4.4.2
deploy_doc "4906a29" v4.5.0
deploy_doc "4bae96e" v4.5.1
deploy_doc "25dee4a" v4.6.0
deploy_doc "7a6c9fa" v4.7.0
deploy_doc "9252a51" v4.8.0
deploy_doc "1366172" v4.8.1
deploy_doc "96d1cfb" v4.8.2
deploy_doc "72aee83" v4.9.0
deploy_doc "bff1c71" v4.9.1
deploy_doc "41981a2" v4.9.2
deploy_doc "39cb6f5" v4.10.0
deploy_doc "28e2787" v4.10.1
deploy_doc "dc193c9" v4.11.0
deploy_doc "54f9d62" v4.11.1
deploy_doc "7655f11" v4.11.2
deploy_doc "65659a2"  # v4.11.3 Latest stable release