// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
const spawn = require('child_process').spawn


  

export default async (req, res) => {
  const {tip,algo} = req.body
  let string
  const tipstring='./pages/api/'+tip+'.csv'
  console.log(tipstring,algo)
  const python = spawn('python', ['./pages/api/sa.py', tipstring, algo])
  python.stdout.on('data', data=>{
    console.log(data)
    string=data.toString()
    res.statusCode = 200
    res.json({ name: JSON.parse(string).name })
  })
  
}
