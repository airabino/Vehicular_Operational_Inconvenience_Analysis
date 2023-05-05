import configparser

def Parse(file='Keys/keys.txt'):

	keys_dict={}

	config=configparser.ConfigParser()
	config.read(file)

	for key in config['Keys']:
		key_in=config['Keys'][key]
		key_in=key_in.replace('"','')
		keys_dict[key]=key_in

	return keys_dict