############ PREPARATION ############
shell:
    poetry run ipython

install-dependencies:
    poetry lock
    poetry install

formatcsv path:
    sed -i '1 s/ /_/g' {{path}}

csv2pkl path:
    #!/usr/bin/env python3
    from pathlib import Path
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    path = Path("{{path}}")
    df = pd.read_csv(path)
    df = df.astype({k: CategoricalDtype(ordered=True) for k, d in df.dtypes.items() if d == "O"})
    df.to_pickle(path.with_suffix(".pkl"))

pkl2table dataset version:
    poetry run python -m lecarb dataset table -d{{dataset}} -v{{version}} --overwrite

table2num dataset version:
    poetry run python -m lecarb dataset dump -d{{dataset}} -v{{version}}

syn2postgres dataset='dom1000' version='skew0.0_corr0.0':
    echo "DROP TABLE IF EXISTS \"{{dataset}}_{{version}}\";" > tmp.sql
    echo "CREATE TABLE \"{{dataset}}_{{version}}\"(" >> tmp.sql
    echo "	col0  DOUBLE PRECISION," >> tmp.sql
    echo "	col1  DOUBLE PRECISION);" >> tmp.sql
    echo "\\\copy \"{{dataset}}_{{version}}\" FROM 'data/{{dataset}}/{{version}}.csv' DELIMITER ',' CSV HEADER;" >> tmp.sql
    $PSQL $DATABASE_URL -f tmp.sql

census2postgres version='original' dataset='census13':
    echo "DROP TABLE IF EXISTS \"{{dataset}}_{{version}}\";" > tmp.sql
    echo "CREATE TABLE \"{{dataset}}_{{version}}\"(" >> tmp.sql
    echo "	age  DOUBLE PRECISION," >> tmp.sql
    echo "	workclass  VARCHAR(64)," >> tmp.sql
    echo "	education  VARCHAR(64)," >> tmp.sql
    echo "	education_num  DOUBLE PRECISION," >> tmp.sql
    echo "	marital_status  VARCHAR(64)," >> tmp.sql
    echo "	occupation  VARCHAR(64)," >> tmp.sql
    echo "	relationship  VARCHAR(64)," >> tmp.sql
    echo "	race  VARCHAR(64)," >> tmp.sql
    echo "	sex  VARCHAR(64)," >> tmp.sql
    echo "	capital_gain  DOUBLE PRECISION," >> tmp.sql
    echo "	capital_loss  DOUBLE PRECISION," >> tmp.sql
    echo "	hours_per_week  DOUBLE PRECISION," >> tmp.sql
    echo "	native_country  VARCHAR(64));" >> tmp.sql
    echo "\\\copy \"{{dataset}}_{{version}}\" FROM 'data/{{dataset}}/{{version}}.csv' DELIMITER ',' CSV HEADER;" >> tmp.sql
    $PSQL $DATABASE_URL -f tmp.sql

census2kdepg version='original' dataset='census13':
    echo "DROP TABLE IF EXISTS \"{{dataset}}_{{version}}\";" > tmp.sql
    echo "CREATE TABLE \"{{dataset}}_{{version}}\"(" >> tmp.sql
    echo "	age  DOUBLE PRECISION," >> tmp.sql
    echo "	workclass  DOUBLE PRECISION," >> tmp.sql
    echo "	education  DOUBLE PRECISION," >> tmp.sql
    echo "	education_num  DOUBLE PRECISION," >> tmp.sql
    echo "	marital_status  DOUBLE PRECISION," >> tmp.sql
    echo "	occupation  DOUBLE PRECISION," >> tmp.sql
    echo "	relationship  DOUBLE PRECISION," >> tmp.sql
    echo "	race  DOUBLE PRECISION," >> tmp.sql
    echo "	sex  DOUBLE PRECISION," >> tmp.sql
    echo "	capital_gain  DOUBLE PRECISION," >> tmp.sql
    echo "	capital_loss  DOUBLE PRECISION," >> tmp.sql
    echo "	hours_per_week  DOUBLE PRECISION," >> tmp.sql
    echo "	native_country  DOUBLE PRECISION);" >> tmp.sql
    echo "\\\copy \"{{dataset}}_{{version}}\" FROM 'data/{{dataset}}/{{version}}_num.csv' DELIMITER ',' CSV HEADER;" >> tmp.sql
    $KDE_PSQL $KDE_DATABASE_URL -f tmp.sql

census2mysql version='original' dataset='census13':
    echo "DROP TABLE IF EXISTS \`{{dataset}}_{{version}}\`;" > tmp.sql
    echo "CREATE TABLE \`{{dataset}}_{{version}}\`(" >> tmp.sql
    echo "	age  DOUBLE PRECISION," >> tmp.sql
    echo "	workclass  VARCHAR(64)," >> tmp.sql
    echo "	education  VARCHAR(64)," >> tmp.sql
    echo "	education_num  DOUBLE PRECISION," >> tmp.sql
    echo "	marital_status  VARCHAR(64)," >> tmp.sql
    echo "	occupation  VARCHAR(64)," >> tmp.sql
    echo "	relationship  VARCHAR(64)," >> tmp.sql
    echo "	race  VARCHAR(64)," >> tmp.sql
    echo "	sex  VARCHAR(64)," >> tmp.sql
    echo "	capital_gain  DOUBLE PRECISION," >> tmp.sql
    echo "	capital_loss  DOUBLE PRECISION," >> tmp.sql
    echo "	hours_per_week  DOUBLE PRECISION," >> tmp.sql
    echo "	native_country  VARCHAR(64));" >> tmp.sql
    echo "SET GLOBAL local_infile = 'ON';" >> tmp.sql
    echo "SHOW GLOBAL VARIABLES LIKE 'local_infile';" >> tmp.sql
    echo "LOAD DATA LOCAL INFILE 'data/{{dataset}}/{{version}}.csv' INTO TABLE \`{{dataset}}_{{version}}\` FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" >> tmp.sql
    $MYSQL --local-infile --protocol tcp -h$MYSQL_HOST --port $MYSQL_PORT -u$MYSQL_USER -p$MYSQL_PSWD $MYSQL_DB < tmp.sql

forest2postgres version='original' dataset='forest10':
    echo "DROP TABLE IF EXISTS \"{{dataset}}_{{version}}\";" > tmp.sql
    echo "CREATE TABLE \"{{dataset}}_{{version}}\"(" >> tmp.sql
    echo "	Elevation  DOUBLE PRECISION," >> tmp.sql
    echo "	Aspect  DOUBLE PRECISION," >> tmp.sql
    echo "	Slope  DOUBLE PRECISION," >> tmp.sql
    echo "	Horizontal_Distance_To_Hydrology  DOUBLE PRECISION," >> tmp.sql
    echo "	Vertical_Distance_To_Hydrology  DOUBLE PRECISION," >> tmp.sql
    echo "	Horizontal_Distance_To_Roadways  DOUBLE PRECISION," >> tmp.sql
    echo "	Hillshade_9am  DOUBLE PRECISION," >> tmp.sql
    echo "	Hillshade_Noon  DOUBLE PRECISION," >> tmp.sql
    echo "	Hillshade_3pm  DOUBLE PRECISION," >> tmp.sql
    echo "	Horizontal_Distance_To_Fire_Points  DOUBLE PRECISION);" >> tmp.sql
    echo "\\\copy \"{{dataset}}_{{version}}\" FROM 'data/{{dataset}}/{{version}}.csv' DELIMITER ',' CSV HEADER;" >> tmp.sql
    $PSQL $DATABASE_URL -f tmp.sql

forest2mysql version='original' dataset='forest10':
    echo "DROP TABLE IF EXISTS \`{{dataset}}_{{version}}\`;" > tmp.sql
    echo "CREATE TABLE \`{{dataset}}_{{version}}\`(" >> tmp.sql
    echo "	Elevation  DOUBLE PRECISION," >> tmp.sql
    echo "	Aspect  DOUBLE PRECISION," >> tmp.sql
    echo "	Slope  DOUBLE PRECISION," >> tmp.sql
    echo "	Horizontal_Distance_To_Hydrology  DOUBLE PRECISION," >> tmp.sql
    echo "	Vertical_Distance_To_Hydrology  DOUBLE PRECISION," >> tmp.sql
    echo "	Horizontal_Distance_To_Roadways  DOUBLE PRECISION," >> tmp.sql
    echo "	Hillshade_9am  DOUBLE PRECISION," >> tmp.sql
    echo "	Hillshade_Noon  DOUBLE PRECISION," >> tmp.sql
    echo "	Hillshade_3pm  DOUBLE PRECISION," >> tmp.sql
    echo "	Horizontal_Distance_To_Fire_Points  DOUBLE PRECISION);" >> tmp.sql
    echo "SET GLOBAL local_infile = 'ON';" >> tmp.sql
    echo "SHOW GLOBAL VARIABLES LIKE 'local_infile';" >> tmp.sql
    echo "LOAD DATA LOCAL INFILE 'data/{{dataset}}/{{version}}.csv' INTO TABLE \`{{dataset}}_{{version}}\` FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" >> tmp.sql
    $MYSQL --local-infile --protocol tcp -h$MYSQL_HOST --port $MYSQL_PORT -u$MYSQL_USER -p$MYSQL_PSWD $MYSQL_DB < tmp.sql

power2postgres version='original' dataset='power7':
    echo "DROP TABLE IF EXISTS \"{{dataset}}_{{version}}\";" > tmp.sql
    echo "CREATE TABLE \"{{dataset}}_{{version}}\"(" >> tmp.sql
    echo "	Global_active_power  DOUBLE PRECISION," >> tmp.sql
    echo "	Global_reactive_power  DOUBLE PRECISION," >> tmp.sql
    echo "	Voltage  DOUBLE PRECISION," >> tmp.sql
    echo "	Global_intensity  DOUBLE PRECISION," >> tmp.sql
    echo "	Sub_metering_1  DOUBLE PRECISION," >> tmp.sql
    echo "	Sub_metering_2  DOUBLE PRECISION," >> tmp.sql
    echo "	Sub_metering_3  DOUBLE PRECISION);" >> tmp.sql
    echo "\\\copy \"{{dataset}}_{{version}}\" FROM 'data/{{dataset}}/{{version}}.csv' DELIMITER ',' CSV HEADER;" >> tmp.sql
    $PSQL $DATABASE_URL -f tmp.sql

power2mysql version='original' dataset='power7':
    echo "DROP TABLE IF EXISTS \`{{dataset}}_{{version}}\`;" > tmp.sql
    echo "CREATE TABLE \`{{dataset}}_{{version}}\`(" >> tmp.sql
    echo "	Global_active_power  DOUBLE PRECISION," >> tmp.sql
    echo "	Global_reactive_power  DOUBLE PRECISION," >> tmp.sql
    echo "	Voltage  DOUBLE PRECISION," >> tmp.sql
    echo "	Global_intensity  DOUBLE PRECISION," >> tmp.sql
    echo "	Sub_metering_1  DOUBLE PRECISION," >> tmp.sql
    echo "	Sub_metering_2  DOUBLE PRECISION," >> tmp.sql
    echo "	Sub_metering_3  DOUBLE PRECISION);" >> tmp.sql
    echo "SET GLOBAL local_infile = 'ON';" >> tmp.sql
    echo "SHOW GLOBAL VARIABLES LIKE 'local_infile';" >> tmp.sql
    echo "LOAD DATA LOCAL INFILE 'data/{{dataset}}/{{version}}.csv' INTO TABLE \`{{dataset}}_{{version}}\` FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" >> tmp.sql
    $MYSQL --local-infile --protocol tcp -h$MYSQL_HOST --port $MYSQL_PORT -u$MYSQL_USER -p$MYSQL_PSWD $MYSQL_DB < tmp.sql

dmv2postgres version='original' dataset='dmv11':
    echo "DROP TABLE IF EXISTS \"{{dataset}}_{{version}}\";" > tmp.sql
    echo "CREATE TABLE \"{{dataset}}_{{version}}\"(" >> tmp.sql
    echo "	Record_Type  VARCHAR(64)," >> tmp.sql
    echo "	Registration_Class  VARCHAR(64)," >> tmp.sql
    echo "	State  VARCHAR(64)," >> tmp.sql
    echo "	County  VARCHAR(64)," >> tmp.sql
    echo "	Body_Type  VARCHAR(64)," >> tmp.sql
    echo "	Fuel_Type  VARCHAR(64)," >> tmp.sql
    echo "	Reg_Valid_Date  DOUBLE PRECISION," >> tmp.sql
    echo "	Color  VARCHAR(64)," >> tmp.sql
    echo "	Scofflaw_Indicator  VARCHAR(64)," >> tmp.sql
    echo "	Suspension_Indicator  VARCHAR(64)," >> tmp.sql
    echo "	Revocation_Indicator  VARCHAR(64));" >> tmp.sql
    echo "\\\copy \"{{dataset}}_{{version}}\" FROM 'data/{{dataset}}/{{version}}.csv' DELIMITER ',' CSV HEADER;" >> tmp.sql
    $PSQL $DATABASE_URL -f tmp.sql

dmv2kdepg version='original' dataset='dmv11':
    echo "DROP TABLE IF EXISTS \"{{dataset}}_{{version}}\";" > tmp.sql
    echo "CREATE TABLE \"{{dataset}}_{{version}}\"(" >> tmp.sql
    echo "	Record_Type  DOUBLE PRECISION," >> tmp.sql
    echo "	Registration_Class  DOUBLE PRECISION," >> tmp.sql
    echo "	State  DOUBLE PRECISION," >> tmp.sql
    echo "	County  DOUBLE PRECISION," >> tmp.sql
    echo "	Body_Type  DOUBLE PRECISION," >> tmp.sql
    echo "	Fuel_Type  DOUBLE PRECISION," >> tmp.sql
    echo "	Reg_Valid_Date  DOUBLE PRECISION," >> tmp.sql
    echo "	Color  DOUBLE PRECISION," >> tmp.sql
    echo "	Scofflaw_Indicator  DOUBLE PRECISION," >> tmp.sql
    echo "	Suspension_Indicator  DOUBLE PRECISION," >> tmp.sql
    echo "	Revocation_Indicator  DOUBLE PRECISION);" >> tmp.sql
    echo "\\\copy \"{{dataset}}_{{version}}\" FROM 'data/{{dataset}}/{{version}}_num.csv' DELIMITER ',' CSV HEADER;" >> tmp.sql
    $KDE_PSQL $KDE_DATABASE_URL -f tmp.sql

dmv2mysql version='original' dataset='dmv11':
    echo "DROP TABLE IF EXISTS \`{{dataset}}_{{version}}\`;" > tmp.sql
    echo "CREATE TABLE \`{{dataset}}_{{version}}\`(" >> tmp.sql
    echo "	Record_Type  VARCHAR(64)," >> tmp.sql
    echo "	Registration_Class  VARCHAR(64)," >> tmp.sql
    echo "	State  VARCHAR(64)," >> tmp.sql
    echo "	County  VARCHAR(64)," >> tmp.sql
    echo "	Body_Type  VARCHAR(64)," >> tmp.sql
    echo "	Fuel_Type  VARCHAR(64)," >> tmp.sql
    echo "	Reg_Valid_Date  DOUBLE PRECISION," >> tmp.sql
    echo "	Color  VARCHAR(64)," >> tmp.sql
    echo "	Scofflaw_Indicator  VARCHAR(64)," >> tmp.sql
    echo "	Suspension_Indicator  VARCHAR(64)," >> tmp.sql
    echo "	Revocation_Indicator  VARCHAR(64));" >> tmp.sql
    echo "SET GLOBAL local_infile = 'ON';" >> tmp.sql
    echo "SHOW GLOBAL VARIABLES LIKE 'local_infile';" >> tmp.sql
    echo "LOAD DATA LOCAL INFILE 'data/{{dataset}}/{{version}}.csv' INTO TABLE \`{{dataset}}_{{version}}\` FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" >> tmp.sql
    $MYSQL --local-infile --protocol tcp -h$MYSQL_HOST --port $MYSQL_PORT -u$MYSQL_USER -p$MYSQL_PSWD $MYSQL_DB < tmp.sql

fetch-quicksel:
    git clone git@github.com:sfu-db/quicksel.git

############ SYNTHETIC DATA GENERATION ############
data-gen skew='0.0' corr='0.0' dom='1000' col='2' seed='123':
    poetry run python -m lecarb dataset gen -s{{seed}} -ddom{{dom}} -vskew{{skew}}_corr{{corr}} --params \
        "{'row_num': 1000000, 'col_num': {{col}}, 'dom': {{dom}}, 'skew': {{skew}}, 'corr': {{corr}}}"

############ QUERY GENERATION ############
wkld-gen-vood data version:
    poetry run python -m lecarb workload gen -d{{data}} -v{{version}} -wvood --params \
        "{'attr': {'pred_number': 1.0}, \
        'center': {'vocab_ood': 1.0}, \
        'width': {'uniform': 0.5, 'exponential': 0.5}, \
        'number': {'train': 100000, 'valid': 10000, 'test': 10000}}" --no-label

wkld-gen-base data version name='base':
    poetry run python -m lecarb workload gen -d{{data}} -v{{version}} -w{{name}} --params \
        "{'attr': {'pred_number': 1.0}, \
        'center': {'distribution': 0.9, 'vocab_ood': 0.1}, \
        'width': {'uniform': 0.5, 'exponential': 0.5}, \
        'number': {'train': 100000, 'valid': 10000, 'test': 10000}}"

wkld-gen-base-sth10 data version seed name:
    poetry run python -m lecarb workload gen -s{{seed}} -d{{data}} -v{{version}} -w{{name}} --params \
        "{'attr': {'pred_number': 1.0}, \
        'center': {'distribution': 0.9, 'vocab_ood': 0.1}, \
        'width': {'uniform': 0.5, 'exponential': 0.5}, \
        'number': {'train': 10000, 'valid': 1000, 'test': 1000}}"

wkld-gen-mth10 data version name='base':
    #!/bin/bash
    for s in $(seq 0 9); do
        just wkld-gen-{{name}}-sth10 {{data}} {{version}} $s {{name}}_$s &
    done

wkld-merge data version name:
    poetry run python -m lecarb workload merge -d{{data}} -v{{version}} -w{{name}}

wkld-label data version workload :
    poetry run python -m lecarb workload label -d{{data}} -v{{version}} -w{{workload}}

wkld-quicksel data version workload count='5':
    poetry run python -m lecarb workload quicksel -d{{data}} -v{{version}} -w{{workload}} --params \
        "{'count': {{count}}}" --overwrite

wkld-dump data version workload:
    poetry run python -m lecarb workload dump -d{{data}} -v{{version}} -w{{workload}}

############ TRAIN CMD ############
train-naru dataset='census13' version='original' layers='4' fc_hiddens='32' embed_size='4' input_encoding='embed' output_encoding='embed' residual='True' warmups='0' sizelimit='0' epochs='100' workload='base' seed='123':
    poetry run python -m lecarb train -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -enaru --params \
        "{'epochs': {{epochs}}, 'input_encoding': '{{input_encoding}}', 'output_encoding': '{{output_encoding}}', \
        'embed_size': {{embed_size}}, 'layers': {{layers}}, 'fc_hiddens': {{fc_hiddens}}, 'residual': {{residual}}, 'warmups': {{warmups}}}" --sizelimit {{sizelimit}}

train-mscn dataset='census13' version='original' workload='base' num_samples='1000' hid_units='16' epochs='200' bs='1024' train_num='100000' sizelimit='0' seed='123':
    poetry run python -m lecarb train -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -emscn --params \
        "{'epochs': {{epochs}}, 'bs': {{bs}}, 'num_samples': {{num_samples}}, 'hid_units': {{hid_units}}, 'train_num': {{train_num}}}" --sizelimit {{sizelimit}}

train-lw-nn dataset='census13' version='original' workload='base' hid_units='128_64_32' bins='200' train_num='10000' bs='32' sizelimit='0' seed='123':
    poetry run python -m lecarb train -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -elw_nn --params \
        "{'epochs': 500, 'bins': {{bins}}, 'hid_units': '{{hid_units}}', 'train_num': {{train_num}}, 'bs': {{bs}}}" --sizelimit {{sizelimit}}

train-lw-nn-update dataset='census13' version='original' workload='base' hid_units='128_64_32' bins='200' train_num='10000' bs='32' sizelimit='0' seed='123' eq='100':
    poetry run python -m lecarb train -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -elw_nn --params \
        "{'epochs': {{eq}}, 'bins': {{bins}}, 'hid_units': '{{hid_units}}', 'train_num': {{train_num}}, 'bs': {{bs}}}" --sizelimit {{sizelimit}}

train-lw-tree dataset='census13' version='original' workload='base' trees='16' bins='200' train_num='10000' sizelimit='0' seed='123':
    poetry run python -m lecarb train -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -elw_tree --params \
        "{'trees': {{trees}}, 'bins': {{bins}}, 'train_num': {{train_num}}}" --sizelimit {{sizelimit}}

train-deepdb dataset='census13' version='original' hdf_sample_size='1000000' rdc_threshold='0.3' ratio_min_instance_slice='0.01' sizelimit='0' workload='base' seed='123':
    poetry run python -m lecarb train -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -edeepdb --params \
        "{'hdf_sample_size': {{hdf_sample_size}}, 'rdc_threshold': {{rdc_threshold}}, 'ratio_min_instance_slice': {{ratio_min_instance_slice}}}" --sizelimit {{sizelimit}}

############ TEST CMD ############
test-naru model psample='2000' dataset='census13' version='original' workload='base' seed='123':
    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -enaru --params \
        "{'psample':{{psample}}, 'model':'{{model}}'}"

test-mscn model dataset='census13' version='original' workload='base' seed='123':
    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -emscn --params \
        "{'model': '{{model}}'}"

test-lw-nn model dataset='census13' version='original' workload='base' use_cache='True' seed='123':
    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -elw_nn --params \
        "{'model': '{{model}}', 'use_cache': {{use_cache}}}"

test-lw-tree model dataset='census13' version='original' workload='base' use_cache='True' seed='123':
    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -elw_tree --params \
        "{'model': '{{model}}', 'use_cache': {{use_cache}}}"

test-deepdb model dataset='census13' version='original' workload='base' seed='123':
    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -edeepdb --params \
        "{'model': '{{model}}'}" --overwrite

############ RUN TRADITIONAL METHODS ############
test-quicksel dataset='census13' version='original' workload='base' train_num='1000' var_num='-1' count='5':
    #!/bin/bash
    # 0. fetch quicksel and compile
    [ ! -d "./quicksel" ] && just fetch-quicksel
    cd quicksel
    make all
    cd -

    # 1. convert workload for quicksel
    just wkld-quicksel {{dataset}} {{version}} {{workload}} {{count}}

    # 2. get number of rows
    export row_num=$(($(< data/{{dataset}}/{{version}}.csv wc -l)-1))
    echo "Dataset {{dataset}} has ${row_num} of rows!"

    # 3. run quicksel
    cd quicksel
    java -Dproject_home=${PWD} \
                        -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar \
                        -Xmx256g -Xms32g edu.illinois.quicksel.experiments.Test quicksel {{dataset}} {{version}} {{workload}} {{train_num}} $row_num {{var_num}}
    cd ..

test-postgres dataset='census13' version='original' workload='base' stat_target='10000' train_version='original' seed='123':
    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -epostgres --params \
        "{'version': '{{train_version}}', 'stat_target': {{stat_target}}}" --overwrite

test-mysql dataset='census13' version='original' workload='base' bucket='1024' train_version='original' seed='123':
    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -emysql --params \
        "{'version': '{{train_version}}', 'bucket': {{bucket}}}" --overwrite

test-sample dataset='census13' version='original' workload='base' ratio='0.015' train_version='original' seed='123':
    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -esample --params \
        "{'version': '{{train_version}}', 'ratio': {{ratio}}}"

test-mhist dataset='census13' version='original' workload='base' num_bins='30000' train_version='original' seed='123':
    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -emhist --params \
        "{'version': '{{train_version}}', 'num_bins': {{num_bins}}}"

test-bayesnet dataset='census13' version='original' workload='base' samples='200' discretize='100' parallelism='50' train_version='original' seed='123':
    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -ebayesnet --params \
        "{'version': '{{train_version}}', 'samples': {{samples}}, 'discretize': {{discretize}}, 'parallelism': {{parallelism}}}"

test-kde dataset='census13' version='original' workload='base' ratio='0.015' train_num='10000' train_version='original' seed='123':
    #!/bin/bash
    $KDE_POSTGRES -D $KDE_PG_DATA -p 5432 >> postgres.log 2>&1 &
    PGPID=$!
    sleep 2
    echo "postgres started pid: ${PGPID}"

    poetry run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -ekde --params \
        "{'version': '{{train_version}}', 'ratio': {{ratio}}, 'train_num': {{train_num}}}"

    kill -9 $PGPID
    sleep 2
    echo "postgres stopped"

############ Update data ############
### Data update/wkld-update
append-data-ind seed='123' dataset='census13' version='original' ap_size='0.2':
    poetry run python -m lecarb dataset update -d{{dataset}} -s{{seed}} -v{{version}} --params \
        "{'type':'ind', 'batch_ratio':{{ap_size}}}"

append-data-cor seed='123' dataset='census13' version='original' ap_size='0.2':
    poetry run python -m lecarb dataset update -d{{dataset}} -s{{seed}} -v{{version}} --params \
        "{'type':'cor', 'batch_ratio':{{ap_size}}}"

append-data-skew seed='123' dataset='census13' version='original' ap_size='0.2':
    poetry run python -m lecarb dataset update -d{{dataset}} -s{{seed}} -v{{version}} --params \
        "{'type':'skew', 'batch_ratio':{{ap_size}}, 'skew_size':'0.0005'}"

wkld-gen-update-base-train-valid seed='123' dataset='census13' version='original' name='base_update' sample_ratio='0.05':
    poetry run python -m lecarb workload gen -s{{seed}} -d{{dataset}} -v{{version}} -w{{name}} --no-label --params \
        "{'attr': {'pred_number': 1.0}, \
        'center': {'distribution': 0.9, 'vocab_ood': 0.1}, \
        'width': {'uniform': 0.5, 'exponential': 0.5}, \
        'number': {'train': 16000, 'valid': 1000, 'test': 0}}"
    poetry run python -m lecarb workload update-label -s{{seed}} -d{{dataset}} -v{{version}} -w{{name}} --sample-ratio={{sample_ratio}}

wkld-gen-update-base-test seed data version name='base':
    poetry run python -m lecarb workload gen -s{{seed}} -d{{data}} -v{{version}} -w{{name}} --params \
        "{'attr': {'pred_number': 1.0}, \
        'center': {'distribution': 0.9, 'vocab_ood': 0.1}, \
        'width': {'uniform': 0.5, 'exponential': 0.5}, \
        'number': {'train': 0, 'valid': 0, 'test': 10000}}"

### Data-driven model update
update-naru model dataset='census13' version='original' workload='base' seed='123' eq='1':
    poetry run python -m lecarb update-train -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -enaru --overwrite --params \
        "{'model':'{{model}}', 'epochs':{{eq}}}"

update-deepdb model dataset='census13' version='original' workload='base' seed='123':
    poetry run python -m lecarb update-train -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -edeepdb --overwrite --params \
        "{'model':'{{model}}'}"

############ Dynamic exp ############
#### MSCN
dynamic-mscn-census13 dataset='census13' version='original' workload='base' update='ind' interval='0.2' train_num='10000' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just train-mscn {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '500' '8' '100' '256' {{train_num}} '0' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-mscn '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-mscn_hid8_sample500_ep100_bs256_10k-{{seed}}' \
                    {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-mscn 'original_base-mscn_hid8_sample500_ep100_bs256_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

dynamic-mscn-forest10 dataset='forest10' version='original' workload='base' update='ind' interval='0.2' train_num='10000' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05' 
    just train-mscn {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '3000' '32' '100' '256' {{train_num}} '0' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-mscn '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-mscn_hid32_sample3000_ep100_bs256_10k-{{seed}}' \
                    {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-mscn 'original_base-mscn_hid32_sample3000_ep100_bs256_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

dynamic-mscn-power7 dataset='power7' version='original' workload='base' update='ind' interval='0.2' train_num='10000' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just train-mscn {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '5000' '64' '100' '256' {{train_num}} '0' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-mscn '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-mscn_hid64_sample5000_ep100_bs256_10k-{{seed}}' \
                    {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-mscn 'original_base-mscn_hid64_sample5000_ep100_bs256_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

dynamic-mscn-dmv11 dataset='dmv11' version='original' workload='base' update='ind' interval='0.2' train_num='10000' seed='123':
    # just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    # just train-mscn {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '10000' '256' '100' '256' {{train_num}} '0' {{seed}}
    # just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    # just test-mscn '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-mscn_hid256_sample10000_ep100_bs256_10k-{{seed}}' \
                    # {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    # just test-mscn 'original_base-mscn_hid256_sample10000_ep100_bs256_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

#### LW
# lw-tree retrain.
dynamic-lw-tree-census13-retrain dataset='census13' version='original' workload='base' update='ind' interval='0.2' train_num='8000' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just census2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just train-lw-tree {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '64' '200' {{train_num}} '0' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-lw-tree '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-lwxgb_tr64_bin200_8k-{{seed}}' \
                        {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'True' {{seed}}
    just test-lw-tree 'original_base-lwxgb_tr64_bin200_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'False' {{seed}}

dynamic-lw-tree-forest10-retrain dataset='forest10' version='original' workload='base' update='ind' interval='0.2' train_num='8000' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just forest2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just train-lw-tree {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '512' '200' {{train_num}} '0' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-lw-tree '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-lwxgb_tr512_bin200_8k-{{seed}}' \
                        {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'True' {{seed}}
    just test-lw-tree 'original_base-lwxgb_tr512_bin200_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'False' {{seed}}

dynamic-lw-tree-power7-retrain dataset='power7' version='original' workload='base' update='ind' interval='0.2' train_num='8000' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just power2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just train-lw-tree {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '256' '200' {{train_num}} '0' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-lw-tree '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-lwxgb_tr256_bin200_8k-{{seed}}' \
                        {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'True' {{seed}}
    just test-lw-tree 'original_base-lwxgb_tr256_bin200_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'False' {{seed}}

dynamic-lw-tree-dmv11-retrain dataset='dmv11' version='original' workload='base' update='ind' interval='0.2' train_num='8000' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just dmv2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just train-lw-tree {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '8192' '200' {{train_num}} '0' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-lw-tree '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-lwxgb_tr8192_bin200_8k-{{seed}}' \
                        {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'True' {{seed}}
    just test-lw-tree 'original_base-lwxgb_tr8192_bin200_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'False' {{seed}}


# lw-nn retrain
dynamic-lw-nn-census13-retrain dataset='census13' version='original' workload='base' update='ind' interval='0.2' train_num='16000' seed='123' eq='500':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just census2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just train-lw-nn-update {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '64_64_64' '200' {{train_num}} '128' '0' {{seed}} {{eq}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-lw-nn '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-lwnn_hid64_64_64_bin200_ep{{eq}}_bs128_16k-{{seed}}' \
                        {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'True' {{seed}}
    just test-lw-nn 'original_base-lwnn_hid64_64_64_bin200_ep500_bs128_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'False' {{seed}}

dynamic-lw-nn-forest10-retrain dataset='forest10' version='original' workload='base' update='ind' interval='0.2' train_num='16000' seed='123' eq='500':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just forest2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just train-lw-nn-update {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '256_256_128_64' '200' {{train_num}} '32' '0' {{seed}} {{eq}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-lw-nn '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-lwnn_hid256_256_128_64_bin200_ep{{eq}}_bs32_16k-{{seed}}' \
                        {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'True' {{seed}}
    just test-lw-nn 'original_base-lwnn_hid256_256_128_64_bin200_ep500_bs32_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'False' {{seed}}

dynamic-lw-nn-power7-retrain dataset='power7' version='original' workload='base' update='ind' interval='0.2' train_num='16000' seed='123' eq='500':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just power2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just train-lw-nn-update {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '512_512_256' '200' {{train_num}} '128' '0' {{seed}} {{eq}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-lw-nn '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-lwnn_hid512_512_256_bin200_ep{{eq}}_bs128_16k-{{seed}}' \
                        {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'True' {{seed}}
    just test-lw-nn 'original_base-lwnn_hid512_512_256_bin200_ep500_bs128_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'False' {{seed}}

dynamic-lw-nn-dmv11-retrain dataset='dmv11' version='original' workload='base' update='ind' interval='0.2' train_num='16000' seed='123' eq='500':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just dmv2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just train-lw-nn-update {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '2048_1024_512_256' '200' {{train_num}} '32' '0' {{seed}} {{eq}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-lw-nn '{{version}}+{{version}}_{{update}}_{{interval}}_train_{{workload}}_{{update}}-lwnn_hid2048_1024_512_256_bin200_ep{{eq}}_bs32_16k-{{seed}}' \
                        {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'True' {{seed}}
    just test-lw-nn 'original_base-lwnn_hid2048_1024_512_256_bin200_ep500_bs32_100k-123' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' 'False' {{seed}}

# Postgres
dynamic-postgres-census13 dataset='census13' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just census2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-postgres {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '10000' '{{version}}+{{version}}_{{update}}_{{interval}}' {{seed}}
    just test-postgres {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '10000' {{version}} {{seed}}
    just test-postgres {{dataset}} {{version}} {{workload}} '10000' {{version}} {{seed}}

dynamic-postgres-forest10 dataset='forest10' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just forest2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-postgres {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '10000' '{{version}}+{{version}}_{{update}}_{{interval}}' {{seed}}
    just test-postgres {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '10000' {{version}} {{seed}}
    just test-postgres {{dataset}} {{version}} {{workload}} '10000' {{version}} {{seed}}

dynamic-postgres-power7 dataset='power7' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just power2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-postgres {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '10000' '{{version}}+{{version}}_{{update}}_{{interval}}' {{seed}}
    just test-postgres {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '10000' {{version}} {{seed}}
    just test-postgres {{dataset}} {{version}} {{workload}} '10000' {{version}} {{seed}}

dynamic-postgres-dmv11 dataset='dmv11' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just dmv2postgres '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-postgres {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '10000' '{{version}}+{{version}}_{{update}}_{{interval}}' {{seed}}
    just test-postgres {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '10000' {{version}} {{seed}}
    just test-postgres {{dataset}} {{version}} {{workload}} '10000' {{version}} {{seed}}

# MySQL
dynamic-mysql-census13 dataset='census13' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just census2mysql '{{version}}' '{{dataset}}'
    just census2mysql '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-mysql {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '1024' {{version}} {{seed}}
    just test-mysql {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '1024' '{{version}}+{{version}}_{{update}}_{{interval}}' {{seed}}
    just test-mysql {{dataset}} {{version}} {{workload}} '1024' {{version}} {{seed}}

dynamic-mysql-forest10 dataset='forest10' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just forest2mysql '{{version}}' '{{dataset}}'
    just forest2mysql '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-mysql {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '1024' {{version}} {{seed}}
    just test-mysql {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '1024' '{{version}}+{{version}}_{{update}}_{{interval}}' {{seed}}
    just test-mysql {{dataset}} {{version}} {{workload}} '1024' {{version}} {{seed}}

dynamic-mysql-power7 dataset='power7' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just power2mysql '{{version}}' '{{dataset}}'
    just power2mysql '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-mysql {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '1024' {{version}} {{seed}}
    just test-mysql {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '1024' '{{version}}+{{version}}_{{update}}_{{interval}}' {{seed}}
    just test-mysql {{dataset}} {{version}} {{workload}} '1024' {{version}} {{seed}}

dynamic-mysql-dmv11 dataset='dmv11' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just dmv2mysql '{{version}}' '{{dataset}}'
    just dmv2mysql '{{version}}+{{version}}_{{update}}_{{interval}}' '{{dataset}}'
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-mysql {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '1024' {{version}} {{seed}}
    just test-mysql {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' '1024' '{{version}}+{{version}}_{{update}}_{{interval}}' {{seed}}
    just test-mysql {{dataset}} {{version}} {{workload}} '1024' {{version}} {{seed}}

# Naru
dynamic-naru-census13 dataset='census13' version='original' workload='base' update='ind' interval='0.2' seed='123'  eq='1':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just update-naru 'original-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123' \
                    {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' {{seed}} {{eq}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-naru '{{version}}+{{version}}_{{update}}_{{interval}}-resmade_hid16,16,16,16_emb8_ep{{eq}}_embedInembedOut_warm0-{{seed}}' \
                    '2000' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-naru 'original-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123' \
                   '2000' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

dynamic-naru-forest10 dataset='forest10' version='original' workload='base' update='ind' interval='0.2' seed='123' eq='1':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just update-naru 'original-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' {{seed}} {{eq}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-naru '{{version}}+{{version}}_{{update}}_{{interval}}-resmade_hid64,64,64,64_emb8_ep{{eq}}_embedInembedOut_warm4000-{{seed}}' \
                    '2000' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-naru 'original-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123' \
                    '2000' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

dynamic-naru-power7 dataset='power7' version='original' workload='base' update='ind' interval='0.2' seed='123' eq='1':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just update-naru 'original-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' {{seed}} {{eq}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-naru '{{version}}+{{version}}_{{update}}_{{interval}}-resmade_hid128,128,128,128,128_emb16_ep{{eq}}_embedInembedOut_warm4000-{{seed}}' \
                    '2000' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-naru 'original-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123' \
                    '2000' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

dynamic-naru-dmv11 dataset='dmv11' version='original' workload='base' update='ind' interval='0.2' seed='123' eq='1':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05'
    just update-naru 'original-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' {{seed}} {{eq}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-naru '{{version}}+{{version}}_{{update}}_{{interval}}-resmade_hid512,512,512,512_emb128_ep{{eq}}_embedInembedOut_warm4000-{{seed}}' \
                    '2000' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-naru 'original-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123' \
                    '2000' {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

# DeepDB
dynamic-deepdb-census13 dataset='census13' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05' '10000'
    just update-deepdb 'original-spn_sample48842_rdc0.4_ms0.01-123' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-deepdb '{{version}}+{{version}}_{{update}}_{{interval}}-spn_sample48842_rdc0.4_ms0.01-{{seed}}' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-deepdb 'original-spn_sample48842_rdc0.4_ms0.01-123' \
                    {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

dynamic-deepdb-forest10 dataset='forest10' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05' '10000'
    just update-deepdb 'original-spn_sample581012_rdc0.4_ms0.005-123' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-deepdb '{{version}}+{{version}}_{{update}}_{{interval}}-spn_sample581012_rdc0.4_ms0.005-{{seed}}' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-deepdb 'original-spn_sample581012_rdc0.4_ms0.005-123' \
                    {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

dynamic-deepdb-power7 dataset='power7' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05' '10000'
    just update-deepdb 'original-spn_sample2075259_rdc0.3_ms0.001-123' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-deepdb '{{version}}+{{version}}_{{update}}_{{interval}}-spn_sample2075259_rdc0.3_ms0.001-{{seed}}' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-deepdb 'original-spn_sample2075259_rdc0.3_ms0.001-123' \
                    {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}

dynamic-deepdb-dmv11 dataset='dmv11' version='original' workload='base' update='ind' interval='0.2' seed='123':
    just append-data-{{update}} {{seed}} {{dataset}} {{version}} {{interval}}
    just wkld-gen-update-base-train-valid {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' '0.05' '10000'
    just update-deepdb 'original-spn_sample1000000_rdc0.2_ms0.001-123' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'train_{{workload}}_{{update}}' {{seed}}
    just wkld-gen-update-base-test {{seed}} {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}'
    just test-deepdb '{{version}}+{{version}}_{{update}}_{{interval}}-spn_sample1000000_rdc0.2_ms0.001-{{seed}}' \
                     {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}
    just test-deepdb 'original-spn_sample1000000_rdc0.2_ms0.001-123' \
                    {{dataset}} '{{version}}+{{version}}_{{update}}_{{interval}}' 'test_{{workload}}_{{update}}' {{seed}}


############ REPORT ############
report-error file dataset='census13':
    poetry run python -m lecarb report -d{{dataset}} --params \
        "{'file': '{{file}}'}"

report-error-dynamic dataset old_new_file new_new_file T update_time:
    poetry run python -m lecarb report-dynamic --params \
        "{'old_new_file': '{{old_new_file}}', 'new_new_file': '{{new_new_file}}', 'T':{{T}}, update_time:{{update_time}}}"

