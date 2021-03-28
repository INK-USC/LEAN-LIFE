const express = require('express')
const app = express()
const port = 3000

const bodyParser = require('body-parser')
const cors = require('cors')

app.use(bodyParser.json())
app.use(cors())

app.get('/', (req, res) => {
	res.send('Hello World!')
})

app.post("/api/login", (req, res,next)=>{
	console.log("/login",req.body)
	setTimeout(()=>{
		if(req.body.username==='jim'){
			res.json({"username": "jim"})
		}else{
			res.status(401).json({err: "unauthorized"})
		}
	}, 300)

})


app.listen(port, () => {
	console.log(`Example app listening at http://localhost:${port}`)
})