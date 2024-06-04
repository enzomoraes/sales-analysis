## Initializing

You should 

```
docker compose up
```

then follow the ETL folders commands

#### Additional Resources:  
[tps-to-csv.jar utility](https://www.ctrl-alt-dev.nl/Projects/TPS-to-CSV/TPS-to-CSV.html)  
[tps-parse repository](https://github.com/ctrl-alt-dev/tps-parse)

## Folders
### ADR Folder:
- The ADR (Architecture Decision Records) folder contains documentation outlining the project's architectural decisions. These records serve as a valuable reference for understanding the project's design and rationale behind key architectural choices.

### ETL Folders:
- ### Extract folder:
    contains the tools necessary to extract the content.  
    
    #### Extracting and Converting TPS Files to CSV  
    Extract the .zip file: Unzip the compressed file using a suitable archive extraction tool.  

    Convert the .TPS file to CSV: Utilize the tps-to-csv.jar utility to transform the extracted TPS file into a CSV format.  
    ```bash
        java -Xmx6g -jar tps-to-csv.jar -s itensvnd.TPS -t itensvnd.csv
        java -Xmx6g -jar tps-to-csv.jar -s produtos.TPS -t produtos.csv
        java -Xmx6g -jar tps-to-csv.jar -s vendas.TPS -t vendas.csv
    ```
        
    > [!NOTE]  
    > Explanation:  
    java: Invokes the Java runtime environment.  
    -Xmx6g: Allocates 6GB of heap memory to the Java process to handle large CSV files.  
    -jar tps-to-csv.jar: Executes the tps-to-csv.jar utility.  
    -s itensvnd.TPS: Specifies the input TPS file named itensvnd.TPS.  
    -t itensvndfull.csv: Defines the output CSV file named itensvndfull.csv.  
    For detailed documentation and usage instructions, refer to the tps-parse repository: https://github.com/topics/tps.  

    #### To populate the database with data from the csv, you must run:  
    ```
    node load-itensvnd.js
    node load-vendas.js
    node load-produtos.js
    ```
    It may take several minutes to complete

- ### Load folder:  
    contains the tools necessary to load the content extracted into a database.  