def configamend(config: dict) -> dict:
    for key in config.keys():
        if config[key] and type(config[key]) == str:
            # if a config value is supposed to be a list but is mistakenly passed as a "stringed list"
            # then this will pass it back to a list
            if config[key][0] == "[" and config[key][-1] == "]":
                config[key] = config[key][1:-1].split(",")
            # provide flexibility in how we provide empty config values
            # for string types, a blank value should also work
            elif config[key] in ["None", "none", "NONE", "", "Null", "null", "NULL"]:
                config[key] = None
    return config
