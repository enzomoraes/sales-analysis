const { Pool } = require('pg');
const fs = require('node:fs');
const path = require('path');
const Papa = require('papaparse');

const insertQuery = `INSERT INTO item_venda (
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
  $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33
);
`;

async function processCSV(filePath, connectionString) {
  const client = new Pool({ connectionString });
  await client.connect();

  for await (const record of lerCSV(filePath)) {
    await client.query(insertQuery, record);
  }
}

function* lerCSV(caminhoDoArquivo) {
  const data = fs.readFileSync(caminhoDoArquivo, { encoding: 'utf-8' });
  // desconsiderando header
  for (const line of data.split('\n').splice(1)) {
    const parsedLine = Papa.parse(line);
    if (!parsedLine.data[0]) continue;
    yield parsedLine.data[0].map(v => {
      if (!v) return null;
      return v.replace('"', '').replace('"', '').replace('\u0000', '');
    });
  }
}

(async () => {
  const connectionString =
    'postgres://postgres:postgres@localhost:5432/postgres';
  const filePath = path.join(__dirname, '..', 'itensvnd.csv');

  try {
    await processCSV(filePath, connectionString);
  } catch (error) {
    console.error('Error processing CSV:', error);
  } finally {
    console.log('exiting process');
    process.exit(0);
  }
})();
