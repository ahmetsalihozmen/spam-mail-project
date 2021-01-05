import Head from 'next/head'
import styles from '../styles/Home.module.css'
import {Button,Form} from 'react-bootstrap'
import React, {useState} from 'react'

export default function Home() {
  const [type,setType] = useState("messages")
  const [algorithm,setAlgo] = useState("MultinomialNB")
  const [row,setRow] = useState("")
  const [string,setString] = useState("")
  const [out,setOut] = useState("")
  
  const connect = () =>{
    console.log(row,string)
    fetch("https://spam-mail-project.herokuapp.com/api/hello",{
      method: 'POST',
        headers:{'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          tip:type,
          algo:algorithm,
          row:row,
          str:string
        })
    })
    .then(response => response.json())
    .then(string =>{
        setOut(JSON.parse(string.json).Acc)
    })
    
  }
  // JSON.parse(string.json).name
  return (
    <div className={styles.container}>
      <Head>
        <title>Spam Mail</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Form>
        <div>File</div>
        <Form.Group >
          <Form.Control as="select" onChange={(event)=>{
            setType(event.target.value)
          }} >
            <option>messages</option>
            <option>spam</option>
            <option>emails</option>
            <option>spamlar</option>
          </Form.Control>
          <div>Algortihm</div>
          <Form.Control as="select" onChange={(event) =>{
            setAlgo(event.target.value)
          }}>
            <option>MultinomialNB</option>
            <option>SVC</option>
            <option>K-Neighbors</option>
            <option>Deep Learning</option>
          </Form.Control>
          <input type='text' onChange={(event) =>{
            setRow(event.target.value)
          }} placeholder="Row"></input>
          <input type='text' onChange={(event) =>{
            setString(event.target.value)
          }} placeholder="String"></input>
          <div className='but'>
          <Button onClick={connect} >
            Run
          </Button>
          </div>
        </Form.Group>
      </Form>
      <div className='res'>
        <div>Matris={out}</div>
        <div>Accuracy Score:</div>
        <div>Standard Deviation</div>
        <div>Recall score for Label 0</div>
        <div>Recall score for Label 1</div>
        <div>Precision score for Label 0</div>
        <div>Precision score for Label 1</div>
        <div>F1 score for Label 0</div>
        <div>F1 score for Label 1</div>
        <div>Result of row:</div>
        <div>Result of input:</div>
        </div>
      <div></div>
      <style jsx>
                {`
                 .but{
                 text-align:center;
                }

                 .res{
                   width:40%;
                   height:200px;
                   background-color:red;
                 }
                `}
            </style>
    </div>
  )
}


