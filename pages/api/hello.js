// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
const spawn = require('child_process').spawn


  

export default async (req, res) => {
  const {tip,algo,row,str} = req.body
  let string
  const tipstring='./pages/api/'+tip+'.csv'
  console.log(tipstring,algo)
  const python = spawn('python', ['./pages/api/mail.py', tipstring, algo, row,str])
  python.stdout.on('data', data=>{
    string=data.toString()
    console.log(string)
    res.statusCode = 200
    res.json({ json: string })
  })
  
}
//name: JSON.parse(string).name 