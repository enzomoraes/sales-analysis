const { Pool } = require('pg');
const fs = require('node:fs');
const path = require('path');

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

  try {
    await client.connect();

    for await (const record of lerCSV(filePath)) {
      const a = await client.query(insertQuery, record);
    }
  } catch (error) {
    console.error('Error processing CSV:', error);
  } finally {
    await client.end();
  }
}

function* lerCSV(caminhoDoArquivo) {
  const data = fs.readFileSync(caminhoDoArquivo, { encoding: 'utf-8' });
  // desconsiderando header
  for (const line of data.split('\n').splice(1)) {
    yield line.split(',').map(v => {
      if (!v) return null;
      return v.replace('"', '').replace('"', '');
    });
  }
}

(async () => {
  const connectionString =
    'postgres://postgres:postgres@localhost:5432/postgres';
  const filePath = path.join(__dirname, '..', 'extract', 'itensvndfull.csv');

  await processCSV(filePath, connectionString);

  console.log('CSV processing completed');
})();
