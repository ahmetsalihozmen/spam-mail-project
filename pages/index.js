import Head from 'next/head'
import styles from '../styles/Home.module.css'
import {Button,Form} from 'react-bootstrap'
import React, {useState} from 'react'

export default function Home() {
  const [type,setType] = useState("messages")
  const [algorithm,setAlgo] = useState("MultinomialNB")
  const [out,setOut] = useState("")
  
  const connect = () =>{
    fetch("http://localhost:3000/api/hello",{
      method: 'POST',
        headers:{'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          tip:type,
          algo:algorithm
        })
    })
    .then(response => response.json())
    .then(string =>{
      setOut(string.name)
    })
    
  }

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
          <Button onClick={connect} >
            Çalıştır
          </Button>
        </Form.Group>
      </Form>
      <div>{out}</div>
      <div></div>
      <style jsx>
                {`
                 text-align:center;
                `}
            </style>
    </div>
  )
}


