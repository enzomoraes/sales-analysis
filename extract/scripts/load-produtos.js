const { Pool } = require('pg');
const fs = require('node:fs');
const path = require('path');
const Papa = require('papaparse');

const insertQuery = `INSERT INTO produtos (
  "Rec No",
  "Codprod",
  "Ativo",
  "Tipo",
  "Codncm",
  "Codcst",
  "Codclf",
  "Localizacao",
  "Descricao",
  "Estqatual",
  "Estqminimo",
  "Estqmaximo",
  "Lkgrupo",
  "Lksecao",
  "Lkfabricante",
  "Lkunidade",
  "Lkreferencia",
  "Pesoliquido",
  "Pesobruto",
  "Iss",
  "Ipi",
  "Icms",
  "Reducaoicms",
  "Substituicao",
  "Descavista",
  "Descaprazo",
  "Descmaximo",
  "Comissao",
  "Pagacomissao",
  "Custonf",
  "Custodesconto",
  "Custoipi",
  "Custofrete",
  "Icmscompra",
  "Custofinanceiro",
  "Custosubst",
  "Custooutro",
  "Custoreal",
  "Custooper",
  "Custoimpfed",
  "Custoicmsvenda",
  "Custocomissao",
  "Margemlucro",
  "Markup",
  "Precosugerido",
  "Precovenda",
  "Percnivel2",
  "Percnivel3",
  "Percnivel4",
  "Preconivel2",
  "Preconivel3",
  "Preconivel4",
  "Lkfornecedor",
  "Dtultcompra",
  "Prultcompra",
  "Qtdcompra",
  "Dtultvenda",
  "Dtaltpreco",
  "Dtacadastro",
  "Precoanter",
  "Resultabc",
  "Datultabc",
  "Foto",
  "Fracao",
  "Qtdedepo",
  "Qtdeloca",
  "Observacao",
  "Orig",
  "Regime",
  "Modbc",
  "Cst",
  "Csosn",
  "Cean",
  "Cofins",
  "Aliqcredito",
  "Tipocalculo",
  "Tipocalculocofinc",
  "Aliquotapis",
  "Aliquotaconfins",
  "Alicotapisreais",
  "Aliquotaconfinsreais",
  "Valordescontovista",
  "Valordescontoprazo",
  "Valordescontomaximo",
  "Qtdpendente",
  "Pquebra",
  "Pvasilhame",
  "Qtdeloja",
  "Iat",
  "Ppt",
  "Qtdereploja",
  "Prclocacao",
  "Pesavel",
  "Validade",
  "Qtdaentregar",
  "Precopromocao",
  "Vldpromocao",
  "Tabloc2",
  "Nometabela2",
  "Tabloc3",
  "Nometabela3",
  "Tabloc4",
  "Nometabela4",
  "Lnktamanho",
  "Lnkcor",
  "Controlado",
  "Psicotropico",
  "Lklistapreco",
  "Registroms",
  "Pedelote",
  "Ultvenda",
  "Pis",
  "Patrimonio",
  "Cstipi",
  "Cstconfins",
  "Codtributacao",
  "Tributoecf",
  "Qtdecaixa",
  "Varejo",
  "Cxvarejo",
  "Codprodinterno",
  "Custoproducao",
  "Lkgeneroitem",
  "Md5",
  "Pauta",
  "Pautoafederal",
  "Valoripi",
  "Tipocalculoipi",
  "Producao",
  "Qdeanterior",
  "Qtderecontagem",
  "Dtaultimacontagem",
  "Predbc",
  "Validadeproduto",
  "Pro:bula"
)
VALUES (
  $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100, $101, $102, $103, $104, $105, $106, $107, $108, $109, $110, $111, $112, $113, $114, $115, $116, $117, $118, $119, $120, $121, $122, $123, $124, $125, $126, $127, $128, $129, $130, $131, $132, $133, $134, $135
)`;

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
  const filePath = path.join(__dirname, '..', 'produtos.csv');

  try {
    await processCSV(filePath, connectionString);
  } catch (error) {
    console.error('Error processing CSV:', error);
  } finally {
    console.log('exiting process');
    process.exit(0);
  }
})();
