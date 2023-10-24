    import { ClerkProvider } from '../../node_modules@clerk/nextjs';
    import { useRouter } from 'next/router';

    function Signin() {
    const router = useRouter();

    const { signIn } = useClerk();

    const handleSignin = async () => {
        await signIn();

        router.push('/dashboard');
    };

    return (
        <ClerkProvider>
        <div>
            <h1>Sign in</h1>
            <button onClick={handleSignin}>Sign in</button>
        </div>
        </ClerkProvider>
    );
    }

    export default Signin;