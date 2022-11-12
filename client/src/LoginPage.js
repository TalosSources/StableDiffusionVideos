import "./LoginPage.css";
import firebase from "./FirebaseBusiness";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Cookies from "js-cookie";

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState();
  const [password, setPassword] = useState();
  const ref = firebase.firestore().collection("users");

  function login() {
    ref.onSnapshot((querySnapshot) => {
      var correctPassword = "";
      querySnapshot.forEach((doc) => {
        // console.log(doc.data().email);
        if (doc.data().email === email) {
          correctPassword = doc.data().password;
        }
      });
      if (password === correctPassword) {
        console.log("Login successful");
        Cookies.set("loggedInUser", email, { expires: 7 }); // 7 days
        navigate("/");
        console.log(Cookies.get("loggedInUser"));
      }
    });
  }

  return (
    <div className="LoginPage">
      <div className="container">
        <label>Email</label>
        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="input"
          type="text"
        />
        <label>Password</label>
        <input
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="input"
          type="password"
        />
        <div className="btnContainer">
          <button className="btn" onClick={login}>
            Login
          </button>
          <button className="btn">
            <a href="/signup">Sign Up</a>
          </button>
        </div>
      </div>
    </div>
  );
}
