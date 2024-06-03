const { Pool } = require('pg');
const fs = require('node:fs');
const path = require('path');

const insertQuery = `INSERT INTO vendas (
  "Rec No",
  "Iditensvnd",
  "Lkvenda",
  "Lkvendedor",
  "Lkcodprod",
  "Sequenciaitemecf",
  "Qtde",
  "Valorunitario",
  "Perdesconto",
  "Desconto",
  "Total",
  "Percomissao",
  "Comissao",
  "Substituicao",
  "Aliqicms",
  "Valoricms",
  "Valoripi",
  "Valoriss",
  "Custounitario",
  "Qtdcancelada",
  "Valorcancelado",
  "Qtdaentregar",
  "Vlrbruto",
  "Datacancelamento",
  "Ippt Paf",
  "Casasdecimaisquantidade",
  "Casasdecimaisvalorunitario",
  "Codigototalizadorparcial",
  "Lote",
  "Lklote",
  "Autoprevsaude",
  "Descespecial",
  "Turno"
)
VALUES (
  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
);
`;

async function processCSV(filePath, connectionString) {
  const client = new Pool({ connectionString });

  try {
    await client.connect();

    for await (const record of lerCSV(filePath)) {
      await client.query(insertQuery, record);
    }
  } catch (error) {
    console.error('Error processing CSV:', error);
  } finally {
    await client.end();
  }
}

function* lerCSV(caminhoDoArquivo) {
  const data = fs.readFileSync(caminhoDoArquivo, { encoding: 'utf-8' });
  for (const line of data.split('\n').splice(0, 1)) {
    yield line.split(',');
  }
}

(async () => {
  const connectionString =
    'postgres://postgres:postgres@localhost:5432/postgres';
  const filePath = path.join(__dirname, '..', 'extract', 'itensvndfull.csv');

  await processCSV(filePath, connectionString);

  console.log('CSV processing completed');
})();
