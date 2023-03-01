import configparser

def Parse(file='keys.txt'):

	keys_dict={}

	# keys_file=open(file)
	# print([line.rstrip() for line in keys_file])
	config=configparser.ConfigParser()
	config.read(file)
	# print(config._dict.__dict__)

	for key in config['Keys']:
		key_in=config['Keys'][key]
		key_in=key_in.replace('"','')
		keys_dict[key]=key_in

	return keys_dict